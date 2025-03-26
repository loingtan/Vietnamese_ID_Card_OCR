from transformers import AutoTokenizer, AutoModelForTokenClassification
from PIL import Image
import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from paddleocr import PaddleOCR, draw_ocr
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from qreader import QReader
import re
import py_vncorenlp
from langdetect import detect, LangDetectException
from transformers import pipeline
from Levenshtein import distance as levenshtein_distance
from ensemble_boxes import weighted_boxes_fusion
corrector = pipeline("text2text-generation",
                     model="bmd1905/vietnamese-correction-v2")


# Select device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# @st.cache_resource(show_spinner="Loading NER model...")
# def load_vncorenlp_model():
#     return py_vncorenlp.VnCoreNLP(
#         save_dir=r'C:\Users\09398\PycharmProjects\VnId-Card\VnCoreNLP',
#         annotators=["ner"]
#     )

# model = load_vncorenlp_model()


# tokenizer1 = AutoTokenizer.from_pretrained(
#     "NlpHUST/ner-vietnamese-electra-base")
# model1 = AutoModelForTokenClassification.from_pretrained(
#     "NlpHUST/ner-vietnamese-electra-base")

# nlp = pipeline("ner", model=model1, tokenizer=tokenizer1)
# tokenizer2 = AutoTokenizer.from_pretrained("NlpHUST/vi-word-segmentation")
# model2 = AutoModelForTokenClassification.from_pretrained(
#     "NlpHUST/vi-word-segmentation")

# nlp2 = pipeline("token-classification", model=model2, tokenizer=tokenizer2)
# example = "Phát biểu tại phiên thảo luận về tình hình kinh tế xã hội của Quốc hội sáng 28/10 , Bộ trưởng Bộ LĐ-TB&XH Đào Ngọc Dung khái quát , tại phiên khai mạc kỳ họp , lãnh đạo chính phủ đã báo cáo , đề cập tương đối rõ ràng về việc thực hiện các chính sách an sinh xã hội"

# ner_results = nlp2(example)
# print(ner_results)


def load_vietnamese_dictionary():
    with open(r'C:\Users\09398\PycharmProjects\VnId-Card\dictionary\dictionaries\hongocduc\words.txt', 'r', encoding='utf-8') as file:
        words = [line.strip() for line in file]
    return words


st.set_page_config(
    page_title="Vietnamese ID Card Scanner",
    page_icon="🆔",
    layout="wide"
)

st.title("Vietnamese ID Card Scanner")
st.write("Upload an image of a Vietnamese ID card for OCR processing.")


@st.cache_resource
def load_paddle_model():
    """Load the pretrained PaddleOCR model for detection"""
    try:
        model = PaddleOCR(det_model_dir="infer_model")
        return model
    except Exception as e:
        st.error(f"Error loading PaddleOCR model: {e}")
        return None


@st.cache_resource
def load_vietocr_model():
    """Load the VietOCR model for text recognition"""
    try:
        config = Cfg.load_config_from_name('vgg_transformer')
        config['cnn']['pretrained'] = True
        config['device'] = device
        predictor = Predictor(config)
        return predictor
    except Exception as e:
        st.error(f"Error loading VietOCR model: {e}")
        return None


@st.cache_resource
def load_yolo_model_for_detect_text():
    """Load YOLO model for text detection"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(r"yolo_detect_text/best.pt")
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None


@st.cache_resource
def load_yolo_model_for_detect_text_v2():
    """Load YOLO model for text detection"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(r"yolo_detect_text/bestv2.pt")
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None


@st.cache_resource
def load_yolo_model():
    """Load YOLO model for ID card detection"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO("corner_detection_model/weight/25_03_25-YOLOv11n-Corner.pt")  # Load custom trained model
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None


paddle_model = load_paddle_model()
# paddle_global = PaddleOCR(use_angle_cls=True, lang='vi',
#                           det=False, rec=True, cls=True)
vietocr_model = load_vietocr_model()
detect_model = load_yolo_model_for_detect_text()
detect_model_v2 = load_yolo_model_for_detect_text_v2()
# Create a QReader instance
qreader = QReader()


def correct_text(text, candidates, threshold=2):
    closest_match = min(
        candidates, key=lambda c: levenshtein_distance(text, c))
    if levenshtein_distance(text, closest_match) <= threshold:
        return closest_match
    return text


def GetInformation(_results):

    # string = '{"ID_number": "09219802508", "Name": "", "Date_of_birth": "", "Gender": "", "Nationality": "", "Place_of_origin": "", "Place_of_residence": "", "ID_number_box": [[208.0, 171.0], [495.0, 177.0], [495.0, 201.0], [208.0, 195.0]]}'
    # result = json.loads(string)

    result = {}
    result['ID_number'] = ''
    result['Name'] = ''
    result['Date_of_birth'] = ''
    result['Gender'] = ''
    result['Nationality'] = ''
    result['Place_of_origin'] = ''
    result['Place_of_residence'] = ''
    result['ID_number_box'] = ''
    regex_dob = r'[0-9][0-9]/[0-9][0-9]'
    regex_residence = r'[0-9][0-9]/[0-9][0-9]/|[0-9]{4,10}|Date|Demo|Dis|Dec|Dale|fer|ting|gical|ping|exp|ver|pate|cond|trị|đến|không|Không|Có|Pat|ter|ity'
    for i, res in enumerate(_results):
        s = res[0]

        if re.search(r'tên|name', s):
            ID_number = _results[i+1] if re.search(r'[0-9][0-9][0-9]', (re.split(r':|[.]|\s+', _results[i+1][0]))[-1].strip(
            )) else (_results[i+2] if re.search(r'[0-9][0-9][0-9]', _results[i+2][0]) else _results[i+3])
            result['ID_number'] = (
                re.split(r':|[.]|\s+', ID_number[0]))[-1].strip()
            result['ID_number_box'] = ID_number[1]
            Name = _results[i+1] if (not re.search(r'[0-9]',
                                     _results[i+1][0])) else _results[i+2]
            result['Name'] = Name[0].title()
            result['Name_box'] = Name[1] if Name[1] else []
            if (result['Date_of_birth'] == ''):
                DOB = _results[i -
                               2] if re.search(regex_dob, _results[i-2][0]) else []
                result['Date_of_birth'] = (
                    re.split(r':|\s+', DOB[0]))[-1].strip() if DOB else ''
                result['Date_of_birth_box'] = DOB[1] if DOB else []
            continue

        if re.search(r'sinh|birth|bith', s) and (not result['Date_of_birth']):
            if re.search(regex_dob, s):
                DOB = _results[i]
            elif re.search(regex_dob, _results[i-1][0]):
                DOB = _results[i-1]
            elif re.search(regex_dob, _results[i+1][0]):
                DOB = _results[i+1]
            else:
                DOB = []
            result['Date_of_birth'] = (
                re.split(r':|\s+', DOB[0]))[-1].strip() if DOB else ''
            result['Date_of_birth_box'] = DOB[1] if DOB else []
            if re.search(r"Việt Nam", _results[i+1][0]):
                result['Nationality'] = 'Việt Nam'
                result['Nationality_box'] = _results[i+1][1]
            continue
        if re.search(r'Giới|Sex', s):
            Gender = _results[i]
            result['Gender'] = 'Nữ' if re.search(
                r'Nữ|nữ', Gender[0]) else 'Nam'
            result['Gender_box'] = Gender[1] if Gender[1] else []
        if re.search(r'Quốc|tịch|Nat', s):
            if (not re.search(r'ty|ing', re.split(r':|,|[.]|ty|tịch', s)[-1].strip()) and (len(re.split(r':|,|[.]|ty|tịch', s)[-1].strip()) >= 3)):
                Nationality = _results[i]
            elif not re.search(r'[0-9][0-9]/[0-9][0-9]/', _results[i+1][0]):
                Nationality = _results[i+1]
            else:
                Nationality = _results[i-1]
            result['Nationality'] = re.split(
                r':|-|,|[.]|ty|[0-9]|tịch', Nationality[0])[-1].strip().title()
            result['Nationality_box'] = Nationality[1] if Nationality[1] else []
            for s in re.split(r'\s+', result['Nationality']):
                if len(s) < 3:
                    result['Nationality'] = re.split(
                        s, result['Nationality'])[-1].strip().title()
            if re.search(r'Nam', result['Nationality']):
                result['Nationality'] = 'Việt Nam'
            continue

        if re.search(r'Quê|origin|ongin|ngin|orging', s):
            PlaceOfOrigin = [_results[i], _results[i+1]
                             ] if not re.search(r'[0-9]{4}', _results[i+1][0]) else []
            if PlaceOfOrigin:
                if len(re.split(r':|;|of|ging|gin|ggong', PlaceOfOrigin[0][0])[-1].strip()) > 2:
                    result['Place_of_origin'] = ((re.split(
                        r':|;|of|ging|gin|ggong', PlaceOfOrigin[0][0]))[-1].strip() + ', ' + PlaceOfOrigin[1][0])
                else:
                    result['Place_of_origin'] = PlaceOfOrigin[1][0]
                result['Place_of_origin_box'] = PlaceOfOrigin[1][1]
            continue
        if re.search(r'Nơi|trú|residence', s):
            vals2 = "" if (
                i+2 > len(_results)-1) else _results[i+2] if len(_results[i+2][0]) > 5 else _results[-1]
            vals3 = "" if (
                i+3 > len(_results)-1) else _results[i+3] if len(_results[i+3][0]) > 5 else _results[-1]
            if ((re.split(r':|;|residence|ence|end', s))[-1].strip() != ''):
                if (vals2 != '' and not re.search(regex_residence, vals2[0])):
                    PlaceOfResidence = [_results[i], vals2]
                elif (vals3 != '' and not re.search(regex_residence, vals3[0])):
                    PlaceOfResidence = [_results[i], vals3]
                elif not re.search(regex_residence, _results[-1][0]):
                    PlaceOfResidence = [_results[i], _results[-1]]
                else:
                    PlaceOfResidence = [_results[-1], []]
            else:
                PlaceOfResidence = [vals2, []] if (vals2 and not re.search(
                    regex_residence, vals2[0])) else [_results[-1], []]
            if PlaceOfResidence[1]:
                result['Place_of_residence'] = re.split(r':|;|residence|sidencs|ence|end', PlaceOfResidence[0][0])[
                    -1].strip() + ' ' + str(PlaceOfResidence[1][0]).strip()
                result['Place_of_residence_box'] = PlaceOfResidence[1][1]
            else:
                result['Place_of_residence'] = PlaceOfResidence[0][0]
                result['Place_of_residence_box'] = PlaceOfResidence[0][1] if PlaceOfResidence else [
                ]
            continue
        elif (i == len(_results)-1):
            if result['Place_of_residence'] == '':
                if not re.search(regex_residence, _results[-1][0]):
                    PlaceOfResidence = _results[-1]
                elif not re.search(regex_residence, _results[-2][0]):
                    PlaceOfResidence = _results[-2]
                else:
                    PlaceOfResidence = []
                result['Place_of_residence'] = PlaceOfResidence[0] if PlaceOfResidence else ''
                result['Place_of_residence_box'] = PlaceOfResidence[1] if PlaceOfResidence else [
                ]
            if result['Gender'] == '':
                result['Gender_box'] = []
            if result['Nationality'] == '':
                result['Nationality_box'] = []
            if result['Name'] == '':
                result['Name_box'] = []
            if result['Date_of_birth'] == '':
                result['Date_of_birth_box'] = []
            if result['Place_of_origin'] == '':
                result['Place_of_origin_box'] = []
        else:
            continue
    return result


def apply_nms(boxes, scores=None, nms_thresh=0.3):
    """Apply Non-Maximum Suppression to eliminate overlapping boxes"""
    if len(boxes) == 0:
        return []

    # If no scores provided, assume all boxes have equal confidence
    if scores is None:
        scores = np.ones(len(boxes))
    EXPEND = 3
    expanded_boxes = boxes.copy()
    for i, box in enumerate(expanded_boxes):
        x1, y1, x2, y2 = box
        expanded_boxes[i] = [max(0, x1-EXPEND), max(0, y1-EXPEND),
                             x2+EXPEND, y2+EXPEND]

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        expanded_boxes.tolist(), scores.tolist(), 0.5, nms_thresh)

    # Return filtered boxes
    filtered_boxes = []
    for idx in indices:
        # OpenCV 4.5.4+ returns a 1D array
        if isinstance(idx, np.ndarray):
            idx = idx.item()
        filtered_boxes.append(boxes[idx])

    return filtered_boxes


def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    # Convert to [x1, y1, x2, y2] format
    x1_1, y1_1 = box1[0]
    x2_1, y2_1 = box1[1]

    x1_2, y1_2 = box2[0]
    x2_2, y2_2 = box2[1]
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


def WarpAndRec(frame, top_left, top_right, bottom_right, bottom_left):
    w, h, cn = frame.shape
    padding = 4.0
    padding = int(padding * w / 640)
    box = []
    # All points are in format [cols, rows]
    pt_A = top_left[0]-padding, top_left[1]-padding
    pt_B = bottom_left[0]-padding, bottom_left[1]+padding
    pt_C = bottom_right[0]+padding, bottom_right[1]+padding
    pt_D = top_right[0]+padding, top_right[1]-padding
    # Here, I have used L2 norm. You can use L1 also.
    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) +
                        ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) +
                        ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                            [0, maxHeight - 1],
                            [maxWidth - 1, maxHeight - 1],
                            [maxWidth - 1, 0]])
    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    matWarped = cv2.warpPerspective(
        frame, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    # cv2.imwrite(fileName, matWarped)

    s = vietocr_model.predict(Image.fromarray(matWarped))
    box.append(pt_A)
    box.append(pt_D)
    box.append(pt_C)
    box.append(pt_B)
    return [s, box]


# def Segmentation(text):
#     exp = ""
#     ner_results = nlp2(text)
#     for e in ner_results:
#         if "##" in e["word"]:
#             exp = exp + e["word"].replace("##", "")
#         elif e["entity"] == "I":
#             exp = exp + "_" + e["word"]
#         else:
#             exp = exp + " " + e["word"]
#     return exp


def extract_entities(ner_data, entity_type):
    results = []
    current_entity = []
    for entry in ner_data:
        for token in entry:
            if token['entity'].startswith('B-') and token['entity'].endswith(entity_type):
                if current_entity:
                    results.append(' '.join(current_entity).replace('_', ''))
                current_entity = [token['word']]
            elif token['entity'].startswith('I-') and token['entity'].endswith(entity_type):
                current_entity.append(token['word'])
        if current_entity:
            results.append(' '.join(current_entity).replace('_', ''))
            current_entity = []
    return results


def extract_field_info(extracted_texts):
    """
    Extract structured information from texts extracted from an ID document.

    Args:
        extracted_texts (list): List of text strings extracted from document

    Returns:
        dict: Structured information with ID fields
    """
    if not extracted_texts or not isinstance(extracted_texts, list):
        return {}

    structured_info = {
        "id_number": None,
        "full_name": None,
        "date_of_birth": None,
        "nationality": None,
        "sex": None,
        "place_of_origin": None,
        "place_of_residence": None,
        "date_of_expiry": None
    }

    try:
        vietnamese_words = load_vietnamese_dictionary()
    except Exception as e:
        print(f"Warning: Could not load Vietnamese dictionary: {e}")
        vietnamese_words = set()

    # Keep a copy of original texts for reference
    original_texts = extracted_texts.copy()

    # Filter Vietnamese texts more robustly
    extracted_texts_vi = []
    for text in extracted_texts:
        try:
            if safe_detect(text) == 'vi':
                extracted_texts_vi.append(text)
        except Exception:
            # Fall back to basic character-based detection if language detection fails
            if any(ord(c) > 127 for c in text) and not text.isdigit():
                extracted_texts_vi.append(text)

    # Extract ID number - look for 9 or 12 digit patterns
    id_pattern = re.compile(r'\b\d{9}(?:\d{3})?\b')
    for text in extracted_texts:
        match = id_pattern.search(text)
        if match:
            structured_info['id_number'] = match.group(0)
            # Remove the ID number from further processing
            extracted_texts_vi = [
                t for t in extracted_texts_vi if match.group(0) not in t]
            break

    # Extract dates first to avoid confusion with address numbers
    date_pattern = re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b')
    dates = []

    for text in extracted_texts:
        matches = date_pattern.findall(text)
        dates.extend(matches)
        # Remove dates from further processing
        for match in matches:
            extracted_texts_vi = [
                t for t in extracted_texts_vi if match not in t]

    # If we don't find formatted dates, try looking for date components
    if not dates:
        # Look for yyyy-mm-dd or dd.mm.yyyy patterns
        alt_date_pattern = re.compile(
            r'\b\d{1,2}[.-]\d{1,2}[.-]\d{4}\b|\b\d{4}[.-]\d{1,2}[.-]\d{1,2}\b')
        for text in extracted_texts:
            matches = alt_date_pattern.findall(text)
            dates.extend(matches)
            # Remove dates from further processing
            for match in matches:
                extracted_texts_vi = [
                    t for t in extracted_texts_vi if match not in t]

    if dates:
        structured_info['date_of_birth'] = dates[0]
        if len(dates) >= 2:
            structured_info['date_of_expiry'] = dates[1]

    # Extract full name - Vietnamese names are typically in uppercase
    # Improved name detection with better filtering
    name_candidates = [
        t for t in extracted_texts_vi
        if t.isupper() and len(t.split()) >= 2 and len(t) > 5 and not any(c.isdigit() for c in t)
    ]

    if name_candidates:
        # Sort by length and prioritize longer names (more likely to be full names)
        name_candidates.sort(key=len, reverse=True)
        structured_info['full_name'] = correct_text(
            name_candidates[0].strip(), vietnamese_words)
        # Remove the identified name from further processing
        extracted_texts_vi = [
            t for t in extracted_texts_vi if t != name_candidates[0]]

    # Extract gender more reliably
    gender_terms = {
        'male': ['nam', 'male', 'Nam', 'MALE', 'NAM', 'Giới tính: Nam'],
        'female': ['nữ', 'female', 'Nữ', 'FEMALE', 'NỮ', 'Giới tính: Nữ']
    }

    gender = None
    gender_texts = []

    for text in extracted_texts:
        text_lower = text.lower()
        if any(term in text_lower for term in gender_terms['male']):
            gender = "Nam"
            gender_texts.append(text)
            break
        elif any(term in text_lower for term in gender_terms['female']):
            gender = "Nữ"
            gender_texts.append(text)
            break

    # If direct detection fails, use the checkIsMale function as fallback
    if not gender and extracted_texts_vi:
        try:
            gender = "Nam" if checkIsMale(extracted_texts_vi) else "Nữ"
        except Exception as e:
            print(f"Warning: Gender detection failed: {e}")

    structured_info['sex'] = gender

    # Remove gender texts from further processing
    for text in gender_texts:
        if text in extracted_texts_vi:
            extracted_texts_vi.remove(text)

    # Default nationality for Vietnamese IDs
    structured_info['nationality'] = "Việt Nam"

    # Remove nationality-related texts
    nationality_terms = ["Việt Nam", "Quốc tịch"]
    for term in nationality_terms:
        extracted_texts_vi = [t for t in extracted_texts_vi if term not in t]

    # Now focus on location extraction with the remaining texts

    # STEP 1: Identify potential address components
    # This includes standalone numbers, address prefixes, and location names

    # Define patterns that might indicate address components
    address_patterns = [
        # Street/house numbers: like "Số 10", "Nhà 15"
        r"(Số|số|Nhà|nhà)\s+\d+",
        # Unit numbers: like "Tổ 5", "Tó 1"
        r"(Tổ|Tó)\s+\d+",
        r"(Thon|thon|Thôn|thôn)\s+\w+",
        # Block/group numbers: like "Khu 3", "Khối 2"
        r"(Khu|khu|Khối|khối|kp|Kp)\s+\d+",
        # Administrative units
        r"(Xã|Phường|Thị trấn|Huyện|Quận|Thành phố|Tỉnh)\s+\w+",
        # Street names
        r"(Đường|đường|Phố|phố)\s+\w+"
    ]

    # Apply address patterns to identify potential address components
    address_components = []
    for text in extracted_texts_vi:
        for pattern in address_patterns:

            if re.search(pattern, text):

                address_components.append(text)
                break

    # STEP 2: Check for text with numbers that might be address parts
    # but don't match standard patterns (e.g., "15B Nguyễn Trãi")
    number_pattern = re.compile(r'\b\d+\w*\b')
    potential_address_with_numbers = []

    for text in extracted_texts_vi:
        if text not in address_components:  # Skip already identified components
            number_matches = number_pattern.findall(text)
            if number_matches:
                # Check if this looks like an ID number (already handled) or a date
                if not any(id_match in text for id_match in [structured_info.get('id_number', '')]):
                    if not any(date in text for date in dates):
                        # This might be an address component with a number
                        potential_address_with_numbers.append(text)

    # Combine identified address components
    address_components.extend(potential_address_with_numbers)

    # STEP 3: Identify remaining location texts that don't have numbers
    # Look for texts that contain location indicators but weren't caught by patterns
    location_indicators = ['xã', 'phường', 'thị xã',
                           'huyện', 'quận', 'thành phố', 'tỉnh']
    location_texts = []

    for text in extracted_texts_vi:
        if text not in address_components:
            text_lower = text.lower()
            if any(indicator in text_lower for indicator in location_indicators) or ',' in text:
                location_texts.append(text)

    # STEP 4: Look for comma-separated address strings (likely full addresses)
    full_addresses = [
        text for text in extracted_texts_vi if ',' in text and len(text) > 10]

    # Sort full addresses by complexity (number of commas)
    full_addresses.sort(key=lambda x: x.count(','), reverse=True)

    # STEP 5: Assemble address components into coherent addresses

    # First check if we have complete addresses already
    origin_set = False
    residence_set = False

    # Check for explicitly labeled addresses
    origin_prefixes = ["quê quán:", "nguyên quán:", "quê:"]
    residence_prefixes = ["nơi thường trú:",
                          "thường trú:", "cư trú:", "địa chỉ:"]

    for text in extracted_texts_vi:
        text_lower = text.lower()
        if not origin_set and any(text_lower.startswith(prefix) for prefix in origin_prefixes):
            structured_info['place_of_origin'] = correct_text(
                text, vietnamese_words)
            origin_set = True
        elif not residence_set and any(text_lower.startswith(prefix) for prefix in residence_prefixes):
            structured_info['place_of_residence'] = correct_text(
                text, vietnamese_words)
            residence_set = True

    # If we have full addresses but no labeled ones, use them
    if full_addresses:
        if not residence_set:
            structured_info['place_of_residence'] = correct_text(
                full_addresses[0], vietnamese_words)
            residence_set = True

        if not origin_set and len(full_addresses) > 1:
            structured_info['place_of_origin'] = correct_text(
                full_addresses[1], vietnamese_words)
            origin_set = True

    # If we still don't have both addresses, try to assemble from components

    # First, try to identify which components might belong together
    # Group components that are likely part of the same address

    # Function to check if two texts might be part of the same address
    def might_be_same_address(text1, text2):
        # If one is a prefix of the other, they may be related
        if text1.lower().startswith('tổ') or text1.lower().startswith('tó'):
            # Check if text2 has location indicators but not already a full address
            return any(ind in text2.lower() for ind in location_indicators) and ',' not in text2
        return False

    # Group potential address components
    address_groups = []
    used_components = set()

    for i, comp1 in enumerate(address_components):
        if comp1 in used_components:
            continue

        group = [comp1]
        used_components.add(comp1)

        for j, comp2 in enumerate(address_components):
            if j != i and comp2 not in used_components:
                if might_be_same_address(comp1, comp2) or might_be_same_address(comp2, comp1):
                    group.append(comp2)
                    used_components.add(comp2)

        if len(group) >= 1:
            address_groups.append(group)

    print(address_groups, "address_groups")
    for comp in address_components:
        if comp not in used_components:
            address_groups.append([comp])

    # Assemble grouped components into addresses
    assembled_addresses = []
    for group in address_groups:
        assembled_addresses.append(", ".join(group))

    # If we have assembled addresses and still need addresses, use them
    print(assembled_addresses)
    if assembled_addresses:
        if residence_set:
            # Sort by length to get the most complete address
            assembled_addresses.sort(key=len, reverse=True)
            structured_info['place_of_residence'] = correct_text(
                assembled_addresses[0], vietnamese_words) + ', ' + structured_info['place_of_residence']
        elif not residence_set:
            # Sort by length to get the most complete address
            assembled_addresses.sort(key=len, reverse=True)
            structured_info['place_of_residence'] = correct_text(
                assembled_addresses[0], vietnamese_words)
            residence_set = True
        if origin_set and len(assembled_addresses) > 1:
            structured_info['place_of_origin'] = correct_text(
                assembled_addresses[1], vietnamese_words) + ', ' + structured_info['place_of_origin']

        elif not origin_set and len(assembled_addresses) > 1:
            structured_info['place_of_origin'] = correct_text(
                assembled_addresses[1], vietnamese_words)
            origin_set = True

    # Special case handling for texts with numbers that might be part of addresses
    if residence_set and potential_address_with_numbers:
        # Look for text pairs that might form an address
        for i, text1 in enumerate(potential_address_with_numbers):
            # Check if this text has numbers and might be a unit number
            if re.search(r'\b\d+\b', text1):
                for j, text2 in enumerate(extracted_texts_vi):
                    if i != j and text2 not in potential_address_with_numbers:
                        # If the second text has location indicators, they might form an address
                        if any(ind in text2.lower() for ind in location_indicators):
                            combined = f"{text1}, {text2}"
                            structured_info['place_of_residence'] = correct_text(
                                combined, vietnamese_words) + ', ' + structured_info['place_of_residence']
                            residence_set = True
                            break
                if residence_set:
                    break

    # Final fallback: if we still don't have both addresses, try using remaining location texts
    location_texts = [t for t in location_texts if t not in full_addresses]
    if location_texts:
        if not residence_set:
            structured_info['place_of_residence'] = correct_text(
                location_texts[0], vietnamese_words)
            residence_set = True

        if not origin_set and len(location_texts) > 1:
            structured_info['place_of_origin'] = correct_text(
                location_texts[1], vietnamese_words)
            origin_set = True

    # Very last resort: if we still don't have residence, look for any text with numbers
    # that might be a partial address and wasn't used elsewhere
    if not residence_set:
        number_texts = [t for t in extracted_texts_vi if re.search(r'\b\d+\b', t)
                        and t not in address_components
                        and t not in gender_texts
                        and structured_info['id_number'] not in t
                        and not any(date in t for date in dates)]

        if number_texts:
            for text in number_texts:
                # Check if this text is likely not a name, ID, date, or gender
                if not text.isupper() and len(text) > 3:
                    structured_info['place_of_residence'] = correct_text(
                        text, vietnamese_words)
                    residence_set = True
                    break

    # Check if we have a residence without an origin
    if residence_set and not origin_set:
        # Try to extract district/province from residence for origin
        residence = structured_info['place_of_residence']
        parts = residence.split(',')
        if len(parts) >= 2:
            # Use the district/province part as origin
            structured_info['place_of_origin'] = correct_text(
                parts[-1].strip(), vietnamese_words)

    return structured_info


def safe_detect(text):
    """
    Safely detect language to prevent crashes.

    Args:
        text (str): Text to detect language for

    Returns:
        str: Detected language code or None
    """
    if not text or not isinstance(text, str):
        return None

    try:
        # Assuming we're using a language detection library like langdetect
        from langdetect import detect
        return detect(text)
    except Exception:
        # Fallback approach - if it contains Vietnamese-specific characters
        vietnamese_chars = set(
            'ăâđêôơưếềểễệốồổỗộớờởỡợứừửữựáàảãạíìỉĩịúùủũụýỳỷỹỵ')
        if any(c.lower() in vietnamese_chars for c in text):
            return 'vi'
        return None


def checkIsMale(texts):
    """
    Determine if the ID belongs to a male based on text cues.

    Args:
        texts (list): List of text strings

    Returns:
        bool: True if likely male, False otherwise
    """
    male_indicators = ['nam', 'Nam', 'NAM',
                       'male', 'Male', 'MALE', 'giới tính: nam']
    female_indicators = ['nữ', 'Nữ', 'NỮ', 'female',
                         'Female', 'FEMALE', 'giới tính: nữ']

    for text in texts:
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in male_indicators):
            return True
        if any(indicator in text_lower for indicator in female_indicators):
            return False

    # Default case - check for common Vietnamese male family names
    # This is a fallback and less reliable
    male_family_names = ['văn', 'hữu', 'đức', 'công', 'quang']

    for text in texts:
        words = text.lower().split()
        if len(words) >= 2 and words[1] in male_family_names:
            return True

    # Default to male if we can't determine (or implement more sophisticated logic)
    return True


def correct_text(text, vietnamese_words):
    """
    Correct common OCR errors in Vietnamese text.

    Args:
        text (str): Text to correct
        vietnamese_words (set): Set of known Vietnamese words

    Returns:
        str: Corrected text
    """
    if not text or not vietnamese_words:
        return text

    # Replace common OCR errors
    replacements = {
        '0': 'O',
        '1': 'I',
        '5': 'S',
        '8': 'B',
        'l': 'I',
        '6': 'G',
    }

    # Remove excessive whitespace
    text = ' '.join(text.split())

    # Apply corrections
    corrected_words = []
    for word in text.split():
        # Skip correction for digits-only words
        if word.isdigit():
            corrected_words.append(word)
            continue

        # Try to correct the word
        for char, replacement in replacements.items():
            if char in word and word not in vietnamese_words:
                candidate = word.replace(char, replacement)
                if candidate in vietnamese_words:
                    word = candidate

        corrected_words.append(word)

    return ' '.join(corrected_words)


def corner_preprocess_image(image, device):
    """Resizes an image to fit within 640x640 while maintaining aspect ratio, then pads it.
    Returns the preprocessed image tensor and scaling factors."""
    h, w, _ = image.shape
    scale = 640 / max(h, w)  # Scale factor to fit within 640x640
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize while keeping aspect ratio
    image_resized = cv2.resize(image, (new_w, new_h))

    # Create a blank 640x640 canvas (black padding)
    padded_image = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Center the resized image
    start_x = (640 - new_w) // 2
    start_y = (640 - new_h) // 2
    padded_image[start_y:start_y + new_h, start_x:start_x + new_w] = image_resized

    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

    return image_tensor, scale, (start_x, start_y)


# def apply_nms(boxes, scores, nms_thresh=0.5):
#     """ Apply NMS to suppress overlapping boxes. """
#     indices = cv2.dnn.NMSBoxes(
#         boxes.tolist(), scores.tolist(), score_threshold=0.4, nms_threshold=nms_thresh)
#     return [boxes[i[0]] for i in indices]


def extract_yolo_results(results, image_shape):
    """ Extract boxes, scores, and class names from YOLO output. """
    boxes, scores, class_ids, class_names = [], [], [], []
    for result in results:

        for box, conf, cls in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy().astype(int)):

            x1, y1, x2, y2 = box

            boxes.append([x1 / image_shape[1], y1 / image_shape[0],
                          x2 / image_shape[1], y2 / image_shape[0]])
            scores.append(float(conf))
            class_ids.append(int(cls))
            class_names.append(result.names[cls])

    return boxes, scores, class_ids, class_names


def draw_yolo(results1, results2, image):
    """ Run YOLO detections, fuse results, draw boxes, and extract text with VietOCR. """
    # Extract results from both YOLO models
    boxes1, scores1, labels1, names1 = extract_yolo_results(
        results1, image.shape)
    boxes2, scores2, labels2, names2 = extract_yolo_results(
        results2, image.shape)

    # Apply Weighted Boxes Fusion
    # Perform weighted boxes fusion on the results from both models
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        [boxes1, boxes2],
        [scores1, scores2],
        [labels1, labels2],
        weights=[1, 1],
        iou_thr=0.5,
        skip_box_thr=0.3
    )

    # x1, y1, x2, y2 = max(0, x1-EXPEND), max(0,
    #                                         y1-EXPEND), x2+EXPEND, y2+EXPEND
    res = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1, y1, x2, y2 = (int(box[0] * image.shape[1]), int(box[1] * image.shape[0]),
                          int(box[2] * image.shape[1]), int(box[3] * image.shape[0]))

        class_name = results1[0].names[label]
        if class_name == "qr":
            continue
        # Draw the rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{class_name} ({score:.2f})"

        # Extract and process text with VietOCR
        cropped_image = image[y1:y2, x1:x2]
        pil_image = Image.fromarray(
            cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        text = vietocr_model.predict(pil_image)

        res.append(text)

    return image, res


def sharpen_image(image):
    """Sharpen the image using an unsharp mask."""
    gaussian_blurred = cv2.GaussianBlur(
        image, (0, 0), 3)  # Apply Gaussian blur
    sharpened = cv2.addWeighted(
        image, 1.5, gaussian_blurred, -0.5, 0)  # Add weighted mask
    return sharpened


def load_model_select(image=None):
    image1 = image.copy()
    image2 = image.copy()

    result1 = detect_model(image1)
    result2 = detect_model_v2(image1)

    vis_image1, info1 = draw_yolo(result1, result2, image1)

    result = paddle_model.ocr(image2, cls=False, det=True, rec=False)
    vis_image2 = draw_ocr(image2, result[0], txts=None, scores=None)
    res = []
    for i, box in enumerate(result[0]):
        top_left = (int(box[0][0]), int(box[0][1]))
        top_right = (int(box[1][0]), int(box[1][1]))
        bottom_right = (int(box[2][0]), int(box[2][1]))
        bottom_left = (int(box[3][0]), int(box[3][1]))
        t = WarpAndRec(image2, top_left, top_right,
                       bottom_right, bottom_left)
        res.append(t)
    info1 = extract_field_info(info1)
    info2 = extract_field_info([t[0] for t in res])
    info = {}

    return info1, vis_image1, vis_image2


# def order_points(pts):
#     """Orders points clockwise starting from top-left: (top-left, top-right, bottom-right, bottom-left)."""
#     rect = np.zeros((4, 2), dtype='float32')

#     # Sort by y-coordinate to get top and bottom points
#     sorted_y = pts[pts[:, 1].argsort()]
#     top_points = sorted_y[:2]  # Two points with smallest y values
#     bottom_points = sorted_y[2:]  # Two points with largest y values

#     # Sort top points by x-coordinate
#     top_points = top_points[top_points[:, 0].argsort()]
#     top_left, top_right = top_points

#     # Sort bottom points by x-coordinate
#     bottom_points = bottom_points[bottom_points[:, 0].argsort()]
#     bottom_left, bottom_right = bottom_points

#     # Arrange in clockwise order starting from top-left
#     rect[0] = top_left     # top-left
#     rect[1] = top_right    # top-right
#     rect[2] = bottom_right # bottom-right
#     rect[3] = bottom_left  # bottom-left

#     return rect


def check_qr_position(image):
    """Check which quadrant contains the QR code and return required rotation degree and the rotated image
        Need to 4 corner image first"""
    height, width = image.shape[:2]
    mid_h, mid_w = height // 2, width // 2

    # Split image into quadrants
    top_left = image[0:mid_h, 0:mid_w]
    top_right = image[0:mid_h, mid_w:width]
    bottom_left = image[mid_h:height, 0:mid_w]
    bottom_right = image[mid_h:height, mid_w:width]

    # Initialize QR code reader
    qreader = QReader()

    # Check each quadrant for QR code
    quadrants = {
        'top_left': top_left,
        'top_right': top_right,
        'bottom_left': bottom_left,
        'bottom_right': bottom_right
    }

    qr_location = None
    for position, quad in quadrants.items():
        qr = qreader.detect_and_decode(quad)
        if qr is not None and len(qr) > 0:
            qr_location = position
            break

    # Determine rotation based on QR location
    if qr_location == 'top_right':
        return image, 0  # Correct orientation
    elif qr_location == 'bottom_left':
        return cv2.rotate(image, cv2.ROTATE_180), 180  # Rotate 180 degrees
    elif qr_location == 'bottom_right': # Rotate left
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 90
    elif qr_location == 'top_left':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), -90  # Rotate right

    return image, 0  # Return original image if no QR code found


def calculate_missed_coord_corner(corners):
    """Calculates the missing fourth corner based on three corners.
    Returns corners in clockwise order starting from top-left."""
    if len(corners) != 3:
        return corners

    # Convert to numpy array
    corners = np.array(corners, dtype='float32')

    # Sort by y-coordinate
    sorted_y = corners[corners[:, 1].argsort()]
    
    # Get top points (smallest y values)
    top_points_mask = corners[:, 1] <= np.median(corners[:, 1])
    top_points = corners[top_points_mask]
    other_points = corners[~top_points_mask]

    if len(top_points) == 2:
        # We have two top points, need to calculate bottom point
        # Sort top points by x
        top_points = top_points[top_points[:, 0].argsort()]
        top_left, top_right = top_points
        
        # Vector from top-left to top-right
        top_vector = top_right - top_left
        
        # Other point is bottom-left or bottom-right
        bottom_point = other_points[0]
        if bottom_point[0] < np.mean([top_left[0], top_right[0]]):
            st.write("Bottom point is bottom-left")
            # Other point is bottom-left, calculate bottom-right
            bottom_left = bottom_point
            bottom_right = bottom_left + top_vector
        else:
            st.write("Bottom point is bottom-right")    
            # Other point is bottom-right, calculate bottom-left
            bottom_right = bottom_point
            bottom_left = bottom_right - top_vector
            
    else:
        # We have one top point and two bottom points
        top_point = top_points[0]
        bottom_points = other_points[bottom_points[:, 0].argsort()]
        
        if len(bottom_points) < 2:
            return corners.tolist()  # Return original corners if we don't have enough points
            
        bottom_left, bottom_right = bottom_points
        
        # Determine if top point is top-left or top-right
        if top_point[0] < np.mean([bottom_left[0], bottom_right[0]]):
            # Top point is top-left, calculate top-right
            st.write("Top point is top-left")
            top_left = top_point
            bottom_vector = bottom_right - bottom_left
            top_right = top_left + bottom_vector
        else:
            st.write("Top point is top-right")
            # Top point is top-right, calculate top-left
            top_right = top_point
            bottom_vector = bottom_right - bottom_left
            top_left = top_right - bottom_vector

    # Return corners in clockwise order starting from top-left
    ordered_corners = np.array([
        top_left,     # top-left
        top_right,    # top-right
        bottom_right, # bottom-right
        bottom_left   # bottom-left
    ])
    
    return ordered_corners.tolist()


def order_points(pts):
    """Orders points in order: (bottom-left, bottom-right, top-left, top-right)."""
    rect = np.zeros((4, 2), dtype='float32')

    # Find top-left and bottom-right using sum coordinates
    s = pts.sum(axis=1)
    temp_tl = pts[np.argmin(s)]  # Temporary top-left
    temp_br = pts[np.argmax(s)]  # Temporary bottom-right

    # Find top-right and bottom-left using difference of coordinates
    diff = np.diff(pts, axis=1)
    temp_tr = pts[np.argmin(diff)]  # Temporary top-right
    temp_bl = pts[np.argmax(diff)]  # Temporary bottom-left

    # Reorder to match [bottom_left, bottom_right, top_left, top_right]
    rect[0] = temp_bl  # bottom-left
    rect[1] = temp_br  # bottom-right
    rect[2] = temp_tl  # top-left
    rect[3] = temp_tr  # top-right

    return rect


def four_point_transform(image, pts):
    """Applies a perspective transform to get a top-down view of the ID."""
    rect = order_points(pts)
    # st.write("Ordered points:", rect)
    (bl, br, tl, tr) = rect  # Now in correct order

    # Compute vertical vector from bottom midpoint to top midpoint
    vertical_vector = (br + bl) / 2 - (tr + tl) / 2  # Vertical direction
    vertical_vector /= np.linalg.norm(vertical_vector)  # Normalize

    rect_extended = np.array([bl, br, tl, tr], dtype='float32')

    # Compute new dimensions
    widthA = np.linalg.norm(br - bl)  # Bottom width
    widthB = np.linalg.norm(tr - tl)  # Top width
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)  # Right height
    heightB = np.linalg.norm(tl - bl)  # Left height
    maxHeight = int(max(heightA, heightB))

    # Define destination points to match our point order
    dst = np.array([
        [0, maxHeight - 1],             # bottom-left
        [maxWidth - 1, maxHeight - 1],  # bottom-right
        [0, 0],                         # top-left
        [maxWidth - 1, 0]               # top-right
    ], dtype='float32')

    # # Debug prints
    # st.write("Input image shape:", image.shape)
    # st.write("Rect extended:", rect_extended)
    # st.write("Destination points:", dst)
    # st.write("Max dimensions:", maxWidth, maxHeight)

    # Compute perspective transform and apply it
    M = cv2.getPerspectiveTransform(rect_extended, dst)
    warped = cv2.warpPerspective(
        image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    # # Debug prints for warped image
    # st.write("Warped image shape:", warped.shape)
    # if warped.size == 0:
    #     st.error("Warning: Warped image is empty!")
    # elif np.all(warped == 0):
    #     st.error("Warning: Warped image is all black!")
    # else:
    #     st.write("Warped image min/max values:", np.min(warped), np.max(warped))

    return warped


def detect_id_card(image, model, device, expand_ratio=0.1):
    """Detects the ID card using YOLO, expands bounding box corners, crops, and corrects orientation."""
    # Get preprocessed image and scaling info
    image_tensor, scale, (pad_x, pad_y) = corner_preprocess_image(image, device)
    results = model(image_tensor)

    # Initialize list to store detected corner points
    corners = []
    for result in results:
        for box in result.boxes.xyxy:
            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, box)
            
            # Calculate center point
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            
            # Convert from padded coordinates back to original image coordinates
            orig_x = (center_x - pad_x) / scale
            orig_y = (center_y - pad_y) / scale
            
            corners.append([int(orig_x), int(orig_y)])

    st.write("Detected corners:", corners)
    if len(corners) <= 2:
        # Try all possible rotations (90, 180, 270 degrees)
        max_corners = corners
        best_image = image
        rotations = [
            (cv2.ROTATE_90_CLOCKWISE, 90),
            (cv2.ROTATE_180, 180),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
        ]

        for rotation_code, angle in rotations:
            # Rotate image
            rotated_image = cv2.rotate(image, rotation_code)
            rotated_tensor, scale, (pad_x, pad_y) = corner_preprocess_image(rotated_image, device)
            rotated_results = model(rotated_tensor)
            
            # Detect corners in rotated image
            rotated_corners = []
            for result in rotated_results:
                for box in result.boxes.xyxy:
                    x_min, y_min, x_max, y_max = map(int, box)
                    center_x = (x_min + x_max) // 2
                    center_y = (y_min + y_max) // 2
                    orig_x = (center_x - pad_x) / scale
                    orig_y = (center_y - pad_y) / scale
                    rotated_corners.append([int(orig_x), int(orig_y)])
            
            if len(rotated_corners) > len(max_corners):
                st.write(f"Found {len(rotated_corners)} corners after {angle}° rotation")
                max_corners = rotated_corners
                best_image = rotated_image

        # Use the rotation that gave us the most corners
        if len(max_corners) > len(corners):
            corners = max_corners
            image = best_image
            st.image(image, caption=f"Rotated image with {len(corners)} corners")
        
        # If we still don't have enough corners, return the original image
        if len(corners) <= 2:
            st.warning("Could not detect enough corners in any orientation")
            return image

    if len(corners) == 3:
        st.write("Found 3 corners at positions:", corners)
        new_corners = calculate_missed_coord_corner(corners)
        if len(new_corners) == 4:
            corners = new_corners
            st.write("Calculated 4th corner. New corners:", corners)
        else:
            st.error(f"Failed to calculate fourth corner. Got {len(new_corners)} corners.")
            st.write("Corner positions:", new_corners)
            return image

    if len(corners) >= 4:
        corners = np.array(corners, dtype="float32")
        
        if len(corners) > 4:
            # Find the best 4 corners that form the largest rectangle
            max_area = 0
            best_corners = None
            
            from itertools import combinations
            for four_corners in combinations(corners, 4):
                four_corners = np.array(four_corners)
                ordered = order_points(four_corners)
                
                # Calculate area using the ordered points
                (bl, br, tl, tr) = ordered
                width1 = np.linalg.norm(tr - tl)   # Top width
                width2 = np.linalg.norm(br - bl)   # Bottom width
                height1 = np.linalg.norm(tr - br)  # Right height
                height2 = np.linalg.norm(tl - bl)  # Left height
                
                # Calculate average area
                area = ((width1 + width2) / 2) * ((height1 + height2) / 2)
                
                # Keep track of largest area without angle verification
                if area > max_area:
                    max_area = area
                    best_corners = ordered

            if best_corners is not None:
                corners = best_corners
                st.write(f"Found largest rectangle with area: {max_area}")
            else:
                st.error("Could not find valid corners")
                return image
        else:
            corners = order_points(corners)

        # Add debug visualization
        debug_img = image.copy()
        for i, corner in enumerate(corners):
            cv2.circle(debug_img, (int(corner[0]), int(corner[1])), 
                      7, (0, 0, 255), -1)
            label = str(i)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(debug_img, label, 
                       (int(corner[0]) - 10, int(corner[1]) - 10),
                       font, 0.8, (255, 255, 255), 2)

        st.image(debug_img, caption="Detected corners (numbered 0-3)")
        
        # Expand corners outward
        center_x, center_y = np.mean(corners, axis=0)
        for i in range(4):
            direction = corners[i] - [center_x, center_y]
            corners[i] += direction * expand_ratio

        # Apply perspective transform
        cropped_id = four_point_transform(image, corners)
        st.image(cropped_id, caption="Cropped and transformed ID card")
        
        # Check QR code position and rotate if necessary
        final_id, rotation_angle = check_qr_position(cropped_id)
        if rotation_angle != 0:
            st.write(f"Image rotated by {rotation_angle} degrees based on QR code position")
            return final_id
        st.image(final_id, caption="Final ID card image")
        return cropped_id

    return image

def process_image(image):
    """Process the image using PaddleOCR and VietOCR"""
    if paddle_model is None:
        st.error("PaddleOCR model not loaded correctly")
        return None
    if vietocr_model is None:
        st.error("VietOCR model not loaded correctly")
        return None

    # Ensure YOLO model is loaded
    yolo_model = load_yolo_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get cropped and rotated image
    processed_image = detect_id_card(image, yolo_model, device)

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

    # Get text detection and recognition results
    info, detected_regions_yolo, detected_regions_db = load_model_select(
        processed_image)

    return {
        "processed_image": processed_image,  # Clean cropped and rotated image
        "detected_regions_yolo": detected_regions_yolo,
        "detected_regions_db": detected_regions_db,  # Image with bounding boxes
        "texts": "",
        "structured_info": info
    }


uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Create two columns for display
    col1, col2, col3, col4 = st.columns(4)

    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display original image
    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 use_container_width=True)

    # Process button
    if st.button("Process ID Card"):
        with st.spinner('Processing image...'):
            result = process_image(image)

            if result:

                with col2:
                    # Display processed image (cropped and rotated)
                    st.subheader("Processed Image")
                    st.image(cv2.cvtColor(
                        result["processed_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

                with col3:
                    # Display image with detected regions
                    st.subheader("Detected regions - YOLO")
                    st.image(cv2.cvtColor(
                        result["detected_regions_yolo"], cv2.COLOR_BGR2RGB), use_container_width=True)

                with col4:
                    # Display image with detected regions
                    st.subheader("Detected regions - DB")
                    st.image(cv2.cvtColor(
                        result["detected_regions_db"], cv2.COLOR_BGR2RGB), use_container_width=True)
                st.subheader("Extracted Information")

                # Display structured information in a table
                for field, value in result["structured_info"].items():
                    if value:
                        st.info(f"**{field}**: {value}")
                    else:
                        st.warning(f"**{field}**: Not detected")

                # # Show all detected text
                # with st.expander("All Detected Text"):
                #     for idx, text in enumerate(result["texts"]):
                #         st.write(f"{idx + 1}. {text}")

                # Option to download results as CSV
                import pandas as pd
                import io

                df = pd.DataFrame([result["structured_info"]])
                csv = df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    "Download Results as CSV",
                    csv,
                    "id_card_results.csv",
                    "text/csv",
                    key='download-csv'
                )
else:
    st.info("Please upload an image to begin processing")

# Add instructions at the bottom
with st.expander("Instructions"):
    st.markdown("""
    ### How to use this application:
    1. Set the path to your pretrained PaddleOCR model in the sidebar
    2. Upload an image of a Vietnamese ID card
    3. Click "Process ID Card" to start OCR
    4. Review the extracted information
    5. Download results as CSV if needed

    ### Note:
    - For best results, ensure the ID card image is centered, well-lit and in focus
    """)
#   - The application works with both old and new Vietnamese ID card formats
#   - Adjust the confidence threshold in the sidebar if needed