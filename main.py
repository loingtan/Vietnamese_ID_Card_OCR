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
st.write("Upload an image of a Vietnamese ID card for OCR processing")


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
def load_yolo_model():
    """Load YOLO model for ID card detection"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO("best.pt")  # Load custom trained model
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
        print(s)
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


def Segmentation(text):
    exp = ""
    ner_results = nlp2(text)
    for e in ner_results:
        if "##" in e["word"]:
            exp = exp + e["word"].replace("##", "")
        elif e["entity"] == "I":
            exp = exp + "_" + e["word"]
        else:
            exp = exp + " " + e["word"]
    return exp


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


def checkIsMale(extracted):
    for text in extracted:
        if "Nữ" in text:
            return False
    return True


def extract_field_info(extracted_texts):
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

    vietnamese_words = load_vietnamese_dictionary()
    known_nationalities = ["Việt Nam"]
    known_sexes = ["nam", "nữ"]

    extracted_texts_vi = [
        text for text in extracted_texts if safe_detect(text) == 'vi']
    # print("vi", extracted_texts_vi)
    # extracted_texts_vi_ner = [nlp(Segmentation(text))
    #                           for text in extracted_texts_vi]
    # combined_text_vi = " ".join(extracted_texts_vi)
    combined_text = " ".join(extracted_texts)

    # person_names = extract_entities(extracted_texts_vi_ner, 'PERSON')
    # print(model.annotate_text(combined_text))
    # locations = extract_entities(extracted_texts_vi_ner, 'LOCATION')
    # print("locations", locations)
    # print("person_names", person_names)
    print(extracted_texts_vi)
    structured_info['id_number'] = next(
        (t for t in extracted_texts if re.match(r'\d{12}', t)), None)
    raw_name = [t for t in extracted_texts_vi if t.isupper() and len(t) > 2]
    extracted_texts_vi = [t for t in extracted_texts_vi if t not in raw_name]
    isMale = checkIsMale(extracted_texts_vi)
    print(isMale)
    raw_name.sort(key=len)
    if raw_name:
        structured_info['full_name'] = correct_text(
            raw_name[0].strip(), vietnamese_words)

    dates = [t for t in extracted_texts if "/" in t]
    print(dates)

    structured_info['date_of_birth'] = dates[0]
    if len(dates) >= 2:
        structured_info['date_of_expiry'] = dates[1]
    raw_nationality = next((t for t in extracted_texts if "Việt" in t), None)

    structured_info['nationality'] = "Việt Nam"
    structured_info['sex'] = "Nam" if isMale else "Nữ"
    locations = ["", ""]
    for t in extracted_texts_vi:
        if len(t.split(",")) >= 2:
            locations[0] = t
            break
    locations[1] = extracted_texts_vi[-2] + ", " + extracted_texts_vi[-1]

    if len(locations) >= 1:
        structured_info['place_of_origin'] = correct_text(
            locations[0], vietnamese_words)
    if len(locations) >= 2:
        structured_info['place_of_residence'] = correct_text(
            locations[1], vietnamese_words)

    return structured_info


def safe_detect(text):
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def corner_preprocess_image(image, device):
    """Automatically adjusts tensor shape for YOLO model input."""
    h, w, _ = image.shape
    size = max(h, w)
    padded_image = np.zeros((size, size, 3), dtype=np.uint8)
    padded_image[:h, :w, :] = image
    image_resized = cv2.resize(padded_image, (640, 640))
    image_tensor = torch.from_numpy(image_resized).permute(
        2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    return image_tensor

def draw_yolo(results, image):
    for result in results:
        # Get top-left and bottom-right corners
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Class IDs
        class_names = [result.names[cls]
                       for cls in class_ids]  # Get class names
        print(boxes, confidences, class_names)
        boxes = apply_nms(boxes, confidences, nms_thresh=0.5)
        info = {}
        for box, conf, class_name in zip(boxes, confidences, class_names):
            x1, y1, x2, y2 = map(int, box)
            if class_name == "qr":
                continue
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_name} ({conf:.2f})"
            cropped_image = image[y1:y2, x1:x2]
            pil_image = Image.fromarray(
                cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            text = vietocr_model.predict(pil_image)
            info[class_name] = text

    return (image, info)


def load_model_select(model="yolo", image=None):
    if model == "yolo":
        result = detect_model(image)
        (vis_image, info) = draw_yolo(result, image)

    elif model == "paddle":
        result = paddle_model.ocr(image, cls=False, det=True, rec=False)
        vis_image = draw_ocr(image, result[0], txts=None, scores=None)
        res = []
        for i, box in enumerate(result[0]):
            top_left = (int(box[0][0]), int(box[0][1]))
            top_right = (int(box[1][0]), int(box[1][1]))
            bottom_right = (int(box[2][0]), int(box[2][1]))
            bottom_left = (int(box[3][0]), int(box[3][1]))
            t = WarpAndRec(image, top_left, top_right,
                           bottom_right, bottom_left)
            res.append(t)
        info = GetInformation(res)
    return info, vis_image

def calculate_missed_coord_corner(corners):
    """Calculates the missing fourth corner based on the other three detected corners."""
    if len(corners) != 3:
        return corners  # Return as is if the length is not 3

    # Convert list to NumPy array for easier manipulation
    corners = np.array(corners, dtype='float32')

    # Calculate the centroid of the three given points
    centroid = np.mean(corners, axis=0)

    # Find the vector from the centroid to each point
    vectors = corners - centroid

    # The missing point should complete the parallelogram, so we assume it lies opposite
    # to the centroid with respect to the sum of the vectors.
    missing_corner = centroid - np.sum(vectors, axis=0)

    # Append the calculated corner to the list
    corners = np.vstack([corners, missing_corner])
    return corners.tolist()


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
        [0, maxHeight - 1],           # bottom-left
        [maxWidth - 1, maxHeight - 1], # bottom-right
        [0, 0],                       # top-left
        [maxWidth - 1, 0]             # top-right
    ], dtype='float32')

    # Compute perspective transform and apply it
    M = cv2.getPerspectiveTransform(rect_extended, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)

    return warped

def check_qr_position(image):
    """Check which quadrant contains the QR code and return required rotation"""
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
    elif qr_location == 'bottom_right':
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 90  # Rotate left
    elif qr_location == 'top_left':
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), -90  # Rotate right
    
    return image, 0  # Return original image if no QR code found

def detect_id_card(image, model, device, expand_ratio=0.1):
    """Detects the ID card using YOLO, expands bounding box corners, crops, and corrects orientation."""
    image_tensor = corner_preprocess_image(image, device)
    results = model(image_tensor)    
    
    # Initialize list to store detected corner points
    corners = []
    for result in results:
        for box in result.boxes.xyxy:
            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, box)
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            corners.append([center_x, center_y])

    # st.write(len(corners))
    
    if len(corners) <= 2:
        return image
    
    # Ensure exactly four corners are detected
    if len(corners) == 3:
        corners = calculate_missed_coord_corner(corners)

    if len(corners) == 4:
        corners = np.array(corners, dtype="float32")
        
        # # Print original corners before ordering
        # st.write("Original corners:")
        # for i, corner in enumerate(corners):
        #     st.write(f"Corner {i}: {corner}")

        # Order corners and print their positions
        ordered_corners = order_points(corners)
        corner_names = ["Bottom-Left", "Bottom-Right", "Top-Left", "Top-Right"]
        # st.write("\nOrdered corners:")
        # for name, corner in zip(corner_names, ordered_corners):
        #     st.write(f"{name}: {corner}")

        # Expand the bounding box corners slightly outward
        center_x, center_y = np.mean(ordered_corners, axis=0)
        for i in range(4):
            direction = ordered_corners[i] - [center_x, center_y]  # Vector from center
            ordered_corners[i] += direction * expand_ratio  # Expand outward

        cropped_id = four_point_transform(image, ordered_corners)
        
        # Check QR code position and rotate if necessary
        final_id, rotation_angle = check_qr_position(cropped_id)
        # if rotation_angle != 0:
        #     st.write(f"Image rotated by {rotation_angle} degrees based on QR code position")
        
        return final_id
    else:
        st.error("Did not detect exactly four corners.")
        return None


def sharpen_image(image):
    """Sharpen the image using an unsharp mask."""
    gaussian_blurred = cv2.GaussianBlur(
        image, (0, 0), 3)  # Apply Gaussian blur
    sharpened = cv2.addWeighted(
        image, 1.5, gaussian_blurred, -0.5, 0)  # Add weighted mask
    return sharpened


def process_image(image, select_model="yolo"):
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
        processed_image = sharpen_image(image)
    else:
        # Resize the processed image
        height, width = processed_image.shape[:2]
        new_width = int(width * 2)
        new_height = int(height * 2)
        processed_image = cv2.resize(processed_image, (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)
        processed_image = sharpen_image(processed_image)

    # Get text detection and recognition results
    info, detected_regions = load_model_select(select_model, processed_image)

    return {
        "processed_image": processed_image,  # Clean cropped and rotated image
        "detected_regions": detected_regions,  # Image with bounding boxes
        "texts": "",
        "structured_info": info
    }


uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])
with st.sidebar:
    detection_model = st.selectbox(
        "Select detection model",
        options=["yolo", "paddle"],
        help="YOLO: Uses YOLO model for text detection\nPaddle: Uses PaddleOCR's DB algorithm"
    )
if uploaded_file is not None:
    # Create two columns for display
    col1, col2, col3 = st.columns(3)

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
            result = process_image(image, detection_model)

            if result:

                with col2:
                    # Display processed image (cropped and rotated)
                    st.subheader("Processed Image")
                    st.image(cv2.cvtColor(
                        result["processed_image"], cv2.COLOR_BGR2RGB), use_container_width=True)
                
                with col3:
                    # Display image with detected regions
                    st.subheader("Detected Regions")
                    st.image(cv2.cvtColor(
                        result["detected_regions"], cv2.COLOR_BGR2RGB), use_container_width=True)

                # Display extracted information
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
    - For best results, ensure the ID card image is clear and well-lit
    - The application works with both old and new Vietnamese ID card formats
    - Adjust the confidence threshold in the sidebar if needed
    """)
