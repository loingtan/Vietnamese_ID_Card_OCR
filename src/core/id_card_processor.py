"""
Core OCR processing pipeline for Vietnamese ID Cards with MongoDB integration.
"""

import cv2
import numpy as np
import json
import io
import re
import time
import logging
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

import torch

# Try relative imports first, fallback to absolute imports for testing
try:
    from ..models.model_manager import ModelManager
    from ..utils.image_processing import (
        apply_nms, calculate_iou, detect_id_card, draw_yolo, sharpen_image, warp_and_recognize,

        pil_to_bytes, resize_image, enhance_image
    )
    from ..utils.text_processing import (
        extract_id_number, extract_dates, extract_gender,
        is_vietnamese_name, normalize_vietnamese_text,
        validate_id_card_fields, load_vietnamese_dictionary
    )
    from ..database.mongodb import db_client
    from ..database.models import OCRResult, ProcessingMetrics, IDCardInfo
    from ..config import get_config
except ImportError:
    from models.model_manager import ModelManager
    from utils.image_processing import (
        apply_nms, calculate_iou, detect_id_card, draw_yolo, sharpen_image, warp_and_recognize,

        pil_to_bytes, resize_image, enhance_image
    )
    from utils.text_processing import (
        extract_id_number, extract_dates, extract_gender,
        is_vietnamese_name, normalize_vietnamese_text,
        validate_id_card_fields, load_vietnamese_dictionary
    )
    # For testing, create mock objects for database
    from types import SimpleNamespace
    db_client = SimpleNamespace()
    db_client.is_connected = False
    OCRResult = dict
    ProcessingMetrics = dict
    IDCardInfo = dict
    from config import get_config

logger = logging.getLogger(__name__)


class IDCardProcessor:
    """Main processor for Vietnamese ID Card OCR."""

    def __init__(self, model_manager: ModelManager):
        self.config = get_config()
        self.model_manager = model_manager
        self.vietnamese_words = load_vietnamese_dictionary()

    def process_image_with_gemini(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process ID card image using Gemini Vision API.

        Args:
            image: PIL Image of the ID card

        Returns:
            Structured information extracted from the ID card
        """
        try:
            client = self.model_manager.get_model('gemini_client')
            if not client:
                return {}

            # Convert PIL Image to PNG bytes
            img_bytes = pil_to_bytes(image, 'PNG')

            # Create content parts for Gemini
            image_part = {
                "inline_data": {
                    "data": img_bytes,
                    "mime_type": "image/png"
                }
            }

            prompt = """
            Analyze this Vietnamese ID card image and extract the following information in JSON format:
            {
                "id_number": "ID number",
                "full_name": "Full name in Vietnamese",
                "date_of_birth": "Date of birth in DD/MM/YYYY format",
                "nationality": "Nationality (usually Việt Nam)",
                "sex": "Gender (Nam/Nữ)",
                "place_of_origin": "Place of origin in Vietnamese",
                "place_of_residence": "Place of residence in Vietnamese",
                "date_of_expiry": "Date of expiry in DD/MM/YYYY format if present"
            }

            Rules:
            1. Return ONLY the JSON object, no other text
            2. If any field is not found, set it to null
            3. Keep Vietnamese text as is, don't translate
            4. Ensure dates are in DD/MM/YYYY format
            5. For names, preserve the exact Vietnamese characters
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=[image_part, prompt]
            )

            # Try to extract JSON from the response
            try:
                return json.loads(response.text)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except:
                        pass
                return {}

        except Exception as e:
            print(f"Error processing image with Gemini: {str(e)}")
            return {}

    def process_image_wtih_vietocr(self, image: Image.Image) -> Dict[str, Any]:
        image1 = image.copy()
        detect_model = self.model_manager._load_yolo_text_detection_model()
        detect_model_v2 = self.model_manager._load_yolo_text_detection_model_v2()
        viet_ocr_model = self.model_manager.get_model('vietocr')
        result1 = detect_model(image1)
        result2 = detect_model_v2(image1)
        vis_image1, info1 = draw_yolo(result1, result2, image1, viet_ocr_model)
        info1 = self.extract_field_info(info1)
        return info1, vis_image1

    def extract_information_from_ocr(self, ocr_results: List) -> Dict[str, Any]:
        """
        Extract structured information from OCR results using rule-based approach.

        Args:
            ocr_results: List of OCR results with text and bounding boxes

        Returns:
            Dictionary with extracted information
        """
        result = {
            'ID_number': '',
            'Name': '',
            'Date_of_birth': '',
            'Gender': '',
            'Nationality': '',
            'Place_of_origin': '',
            'Place_of_residence': '',
            'ID_number_box': '',
            'Name_box': [],
            'Date_of_birth_box': [],
            'Gender_box': [],
            'Nationality_box': [],
            'Place_of_origin_box': [],
            'Place_of_residence_box': []
        }

        regex_dob = r'[0-9][0-9]/[0-9][0-9]'
        regex_residence = r'[0-9][0-9]/[0-9][0-9]/|[0-9]{4,10}|Date|Demo|Dis|Dec|Dale|fer|ting|gical|ping|exp|ver|pate|cond|trị|đến|không|Không|Có|Pat|ter|ity'

        for i, res in enumerate(ocr_results):
            text = res[0]

            # Extract name and ID number
            if re.search(r'tên|name', text, re.IGNORECASE):
                # Look for ID number in next few results
                for j in range(1, min(4, len(ocr_results) - i)):
                    next_res = ocr_results[i + j]
                    id_match = re.search(r'[0-9]{9,12}', next_res[0])
                    if id_match:
                        result['ID_number'] = id_match.group()
                        result['ID_number_box'] = next_res[1]
                        break

                # Look for name (non-numeric text)
                for j in range(1, min(3, len(ocr_results) - i)):
                    next_res = ocr_results[i + j]
                    if not re.search(r'[0-9]', next_res[0]) and is_vietnamese_name(next_res[0]):
                        result['Name'] = next_res[0].title()
                        result['Name_box'] = next_res[1]
                        break

                # Look for date of birth nearby
                if not result['Date_of_birth']:
                    for j in range(-2, 3):
                        if 0 <= i + j < len(ocr_results):
                            check_res = ocr_results[i + j]
                            if re.search(regex_dob, check_res[0]):
                                dob_text = re.split(
                                    r':|\s+', check_res[0])[-1].strip()
                                result['Date_of_birth'] = dob_text
                                result['Date_of_birth_box'] = check_res[1]
                                break
                continue

            # Extract date of birth
            if re.search(r'sinh|birth|bith', text, re.IGNORECASE) and not result['Date_of_birth']:
                if re.search(regex_dob, text):
                    dob_res = ocr_results[i]
                elif i > 0 and re.search(regex_dob, ocr_results[i-1][0]):
                    dob_res = ocr_results[i-1]
                elif i < len(ocr_results) - 1 and re.search(regex_dob, ocr_results[i+1][0]):
                    dob_res = ocr_results[i+1]
                else:
                    dob_res = None

                if dob_res:
                    result['Date_of_birth'] = re.split(
                        r':|\s+', dob_res[0])[-1].strip()
                    result['Date_of_birth_box'] = dob_res[1]

                # Check for nationality nearby
                if i < len(ocr_results) - 1 and re.search(r"Việt Nam", ocr_results[i+1][0]):
                    result['Nationality'] = 'Việt Nam'
                    result['Nationality_box'] = ocr_results[i+1][1]
                continue

            # Extract gender
            if re.search(r'Giới|Sex', text, re.IGNORECASE):
                gender_text = extract_gender(text)
                if gender_text:
                    result['Gender'] = gender_text
                    result['Gender_box'] = res[1]
                continue

            # Extract nationality
            if re.search(r'Quốc|tịch|Nat', text, re.IGNORECASE):
                # Try current result first
                nationality_text = re.split(
                    r':|,|[.]|ty|tịch', text)[-1].strip()
                if len(nationality_text) >= 3 and not re.search(r'ty|ing', nationality_text):
                    nationality_res = ocr_results[i]
                elif i < len(ocr_results) - 1 and not re.search(r'[0-9][0-9]/[0-9][0-9]/', ocr_results[i+1][0]):
                    nationality_res = ocr_results[i+1]
                elif i > 0:
                    nationality_res = ocr_results[i-1]
                else:
                    nationality_res = None

                if nationality_res:
                    nationality = re.split(
                        r':|-|,|[.]|ty|[0-9]|tịch', nationality_res[0])[-1].strip().title()
                    # Clean up nationality text
                    words = nationality.split()
                    nationality = ' '.join([w for w in words if len(w) >= 3])
                    if 'Nam' in nationality:
                        nationality = 'Việt Nam'
                    result['Nationality'] = nationality
                    result['Nationality_box'] = nationality_res[1]
                continue

            # Extract place of origin
            if re.search(r'Quê|origin|ongin|ngin|orging', text, re.IGNORECASE):
                origin_texts = []
                if i < len(ocr_results) - 1 and not re.search(r'[0-9]{4}', ocr_results[i+1][0]):
                    origin_texts = [ocr_results[i], ocr_results[i+1]]

                if origin_texts:
                    origin_part1 = re.split(
                        r':|;|of|ging|gin|ggong', origin_texts[0][0])[-1].strip()
                    if len(origin_part1) > 2:
                        result['Place_of_origin'] = f"{origin_part1}, {origin_texts[1][0]}"
                    else:
                        result['Place_of_origin'] = origin_texts[1][0]
                    result['Place_of_origin_box'] = origin_texts[1][1]
                continue

            # Extract place of residence
            if re.search(r'Nơi|trú|residence', text, re.IGNORECASE):
                residence_candidates = []

                # Check next 2-3 results
                for j in range(2, 4):
                    if i + j < len(ocr_results) and len(ocr_results[i + j][0]) > 5:
                        residence_candidates.append(ocr_results[i + j])

                # Find the best candidate (not matching exclusion patterns)
                residence_res = None
                for candidate in residence_candidates:
                    if not re.search(regex_residence, candidate[0]):
                        residence_res = candidate
                        break

                if not residence_res and ocr_results:
                    # Fallback to last result if it's not excluded
                    if not re.search(regex_residence, ocr_results[-1][0]):
                        residence_res = ocr_results[-1]

                if residence_res:
                    residence_prefix = re.split(
                        r':|;|residence|ence|end', text)[-1].strip()
                    if residence_prefix:
                        result['Place_of_residence'] = f"{residence_prefix} {residence_res[0].strip()}"
                    else:
                        result['Place_of_residence'] = residence_res[0]
                    result['Place_of_residence_box'] = residence_res[1]
                continue

        # Fill in missing box coordinates for empty fields
        for field in ['Gender_box', 'Nationality_box', 'Name_box', 'Date_of_birth_box',
                      'Place_of_origin_box', 'Place_of_residence_box']:
            if not result[field]:
                result[field] = []

        return result

    def extract_field_info(self, extracted_texts: List[str]) -> Dict[str, Any]:
        """
        Extract structured information from a list of extracted texts.

        Args:
            extracted_texts: List of text strings from OCR

        Returns:
            Dictionary with structured ID card information
        """
        if not extracted_texts or not isinstance(extracted_texts, list):
            return {}

        structured_info = {
            "id_number": None,
            "full_name": None,
            "date_of_birth": None,
            "nationality": "Việt Nam",  # Default for Vietnamese IDs
            "sex": None,
            "place_of_origin": None,
            "place_of_residence": None,
            "date_of_expiry": None
        }

        # Extract ID number
        for text in extracted_texts:
            id_num = extract_id_number(text)
            if id_num:
                structured_info['id_number'] = id_num
                break

        # Extract dates
        all_dates = []
        for text in extracted_texts:
            dates = extract_dates(text)
            all_dates.extend(dates)

        if all_dates:
            structured_info['date_of_birth'] = all_dates[0]
            if len(all_dates) >= 2:
                structured_info['date_of_expiry'] = all_dates[1]

        # Extract name - look for Vietnamese names
        for text in extracted_texts:
            if is_vietnamese_name(text):
                structured_info['full_name'] = normalize_vietnamese_text(text)
                break

        # Extract gender
        for text in extracted_texts:
            gender = extract_gender(text)
            if gender:
                structured_info['sex'] = gender
                break

        # Extract addresses (simplified approach)
        address_texts = [text for text in extracted_texts
                         if ',' in text and len(text) > 10 and
                         not any(d in text for d in all_dates)]

        if address_texts:
            # Sort by complexity (more commas = more detailed address)
            address_texts.sort(key=lambda x: x.count(','), reverse=True)

            if len(address_texts) >= 2:
                structured_info['place_of_residence'] = address_texts[0]
                structured_info['place_of_origin'] = address_texts[1]
            elif len(address_texts) == 1:
                structured_info['place_of_residence'] = address_texts[0]

        # Validate and clean the extracted data
        return validate_id_card_fields(structured_info)

    def process_id_card(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Main processing pipeline for ID card images.

        Args:
            image: Input image as numpy array

        Returns:
            Dictionary with processing results
        """
        try:
            # Resize image if too large
            processed_image = resize_image(image)

            # Enhance image quality
            enhanced_image = enhance_image(processed_image)
            yolo_models = self.model_manager._load_yolo_corner_detection_model()
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            processed_image = detect_id_card(
                enhanced_image, yolo_models, device)

            if processed_image is None:
                height, width = image.shape[:2]
                new_width = int(width * 1.8)
                new_height = int(height * 1.8)
                processed_image = cv2.resize(image, (new_width, new_height),
                                             interpolation=cv2.INTER_LINEAR)
                processed_image = sharpen_image(image)
            else:
                # Resize the processed image
                height, width = processed_image.shape[:2]
                new_width = int(width * 2.2)
                new_height = int(height * 2.2)
                processed_image = cv2.resize(processed_image, (new_width, new_height),
                                             interpolation=cv2.INTER_LINEAR)
                processed_image = sharpen_image(processed_image)
            # pil_image = Image.fromarray(cv2.cvtColor(
            #     processed_image, cv2.COLOR_BGR2RGB))

            # Try Gemini processing first (if available)
            info, image = self.process_image_wtih_vietocr(processed_image)
            # gemini_result = self.process_image_with_gemini(pil_image)

            # if gemini_result and any(gemini_result.values()):
            #     return {
            #         'status': 'success',
            #         'method': 'gemini',
            #         'extracted_info': validate_id_card_fields(gemini_result)
            #     }

            # Fallback to traditional OCR pipeline
            # This would involve corner detection, perspective correction, text detection, etc.
            # For now, returning a placeholder
            return {
                'status': 'success',
                'method': 'traditional_ocr',
                'extracted_info': info,
                'message': 'Traditional OCR pipeline not fully implemented yet'
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'extracted_info': {}
            }

    def process_image_with_database(self, image: Image.Image, session_id: str,
                                    filename: str = "uploaded_image.jpg") -> Dict[str, Any]:
        """
        Process ID card image and save results to database.

        Args:
            image: PIL Image of the ID card
            session_id: User session ID
            filename: Original filename

        Returns:
            Complete processing results with database IDs
        """
        start_time = time.time()

        try:
            # Resize image if needed
            img_array = np.array(image)
            img_array = resize_image(img_array)
            image = Image.fromarray(img_array)

            # Process the image
            results = self.process_image(image)
            processing_time = time.time() - start_time

            # Create structured ID card info
            id_card_info = IDCardInfo.from_dict(
                results.get('extracted_info', {}))

            # Create OCR result object
            ocr_result = OCRResult(
                session_id=session_id,
                image_filename=filename,
                extracted_info=results.get('extracted_info', {}),
                processing_time=processing_time,
                confidence_scores=results.get('confidence_scores', {}),
                detected_text_regions=results.get('detected_regions', []),
                qr_code_data=results.get('qr_code_data'),
                gemini_response=results.get('gemini_response'),
                success=True
            )

            # Save to database if connected
            result_id = None
            if db_client.is_connected:
                try:
                    result_id = db_client.save_ocr_result(ocr_result)

                    # Save processing metrics
                    metrics = ProcessingMetrics(
                        operation="ocr_processing",
                        processing_time=processing_time,
                        success=True,
                        session_id=session_id,
                        image_size=(img_array.shape[1], img_array.shape[0]),
                        confidence_score=results.get(
                            'confidence_scores', {}).get('overall', 0.0)
                    )
                    db_client.save_metrics(metrics)

                except Exception as e:
                    logger.error(f"Failed to save to database: {e}")

            # Add database info to results
            results['database_id'] = result_id
            results['processing_time'] = processing_time
            results['completeness_score'] = id_card_info.get_completeness_score()
            results['is_valid'] = id_card_info.is_valid()

            return results

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)

            logger.error(f"Error processing image: {error_msg}")

            # Save error to database
            if db_client.is_connected:
                try:
                    ocr_result = OCRResult(
                        session_id=session_id,
                        image_filename=filename,
                        extracted_info={},
                        processing_time=processing_time,
                        confidence_scores={},
                        detected_text_regions=[],
                        success=False,
                        error_message=error_msg
                    )
                    db_client.save_ocr_result(ocr_result)

                    # Save error metrics
                    metrics = ProcessingMetrics(
                        operation="ocr_processing",
                        processing_time=processing_time,
                        success=False,
                        session_id=session_id,
                        error_type=type(e).__name__,
                        error_message=error_msg
                    )
                    db_client.save_metrics(metrics)

                except Exception as db_error:
                    logger.error(
                        f"Failed to save error to database: {db_error}")

            return {
                'success': False,
                'error': error_msg,
                'processing_time': processing_time
            }

    def get_session_results(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get all OCR results for a session.

        Args:
            session_id: User session ID

        Returns:
            List of OCR results from database
        """
        if not db_client.is_connected:
            return []

        try:
            return db_client.get_ocr_results_by_session(session_id)
        except Exception as e:
            logger.error(f"Failed to get session results: {e}")
            return []

    def search_by_id_number(self, id_number: str) -> List[Dict[str, Any]]:
        """
        Search OCR results by ID number.

        Args:
            id_number: Vietnamese ID number to search for

        Returns:
            List of matching OCR results
        """
        if not db_client.is_connected:
            return []

        try:
            return db_client.search_by_id_number(id_number)
        except Exception as e:
            logger.error(f"Failed to search by ID number: {e}")
            return []
