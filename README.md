# Vietnamese ID Card OCR

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-brightgreen.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project implements an Optical Character Recognition (OCR) system specifically designed to extract information from Vietnamese ID cards (both old and new formats). It utilizes a combination of deep learning models for robust detection and recognition, presented through an interactive Streamlit web application.

The system performs the following key steps:
1.  **ID Card Detection & Alignment:** Detects the ID card in an input image using a YOLO model trained to find corners.
2.  **Perspective Correction:** Warps the detected ID card region to obtain a top-down, rectangular view.
3.  **Orientation Correction:** Uses QR code detection (if present) to ensure the card is correctly oriented.
4.  **Text Detection:** Employs both YOLO and PaddleOCR's detection model (DB) to locate text regions within the aligned ID card image. Results are fused using Weighted Boxes Fusion (WBF) for improved accuracy.
5.  **Text Recognition:** Uses the VietOCR library (VGG-Transformer) to recognize the Vietnamese text within the detected regions.
6.  **Information Extraction:** Parses the recognized text using regular expressions and heuristics to extract key fields like ID number, name, date of birth, gender, nationality, place of origin, and place of residence.
7.  **QR Code Decoding:** Detects and decodes the QR code present on newer ID cards.
8.  **User Interface:** Provides a simple Streamlit interface for uploading ID card images and viewing the extracted results.

## Features

*   Supports both old and new Vietnamese ID card formats.
*   Automatic perspective and orientation correction.
*   Robust text detection using YOLO and DB model fusion.
*   High-accuracy Vietnamese text recognition with VietOCR.
*   Structured extraction of key ID card fields.
*   QR code detection and decoding.
*   Interactive web interface powered by Streamlit.
*   Utilizes GPU acceleration if available (PyTorch/PaddlePaddle).

## Technology Stack

*   **Programming Language:** Python 3.9+
*   **Core Libraries:**
    *   OpenCV (`opencv-python`): Image processing, perspective transform, drawing.
    *   PyTorch: Backend for YOLO and VietOCR models.
    *   Ultralytics YOLO: ID card corner detection and text detection.
    *   PaddlePaddle (`paddlepaddle`): Backend for PaddleOCR.
    *   PaddleOCR (`paddleocr`): Text detection (DB model).
    *   VietOCR (`vietocr`): Vietnamese text recognition.
    *   Streamlit: Web application framework.
    *   NumPy: Numerical operations.
    *   QReader (`qreader`): QR code detection and decoding.
    *   Ensemble-Boxes (`ensemble-boxes`): Weighted Boxes Fusion for detection results.
    *   Transformers (`transformers`): (Optional, used for text correction model)
    *   Levenshtein (`python-Levenshtein`): String similarity for text correction.
*   **Models:**
    *   Custom YOLOv11n: For ID card corner detection.
    *   Custom YOLO: For text field detection.
    *   PaddleOCR DB: For text field detection.
    *   VietOCR VGG-Transformer: For Vietnamese text recognition.
    *   `bmd1905/vietnamese-correction-v2`: (Optional) For text correction.

## Project Structure (Simplified)

```
VnId-Card/
├── main.py                 # Main Streamlit application script
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── requirements_windows.txt# Python dependencies for Windows
├── corner_detection_model/ # YOLO model for corner detection
│   └── weight/
│       └── *.pt
├── infer_model/            # PaddleOCR detection model files
│   ├── *.pdiparams
│   ├── *.pdiparams.info
│   ├── *.pdmodel
│   └── *.yml
├── dictionary/             # Dictionary files for text correction/validation
│   └── dictionaries/
│       └── hongocduc/
│           └── words.txt
└── ... (other potential utility scripts/folders)
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd VnId-Card
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    venv\Scripts\activate
    # Linux/macOS:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Choose the requirements file based on your operating system. Ensure you have the necessary build tools (like C++ compilers) installed, as some libraries might require compilation. If using CUDA for GPU acceleration, ensure you have a compatible PyTorch/PaddlePaddle version installed first, matching your CUDA toolkit version.

    *   **For Windows:**
        ```bash
        pip install -r requirements_windows.txt
        ```
    *   **For others:**
        ```bash
        pip install -r requirements.txt
        ```
    *   *(Note: You might need to install PyTorch/PaddlePaddle with CUDA support separately if not included or if you need a specific version. Refer to their official websites.)*

4.  **Model Files:** The required model files seem to be included in the repository (`corner_detection_model`, `infer_model`, `yolo_detect_text`). Ensure they are correctly placed.

## Usage

1.  **Activate your virtual environment** (if you created one).
2.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```
3.  Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
4.  Upload an image of a Vietnamese ID card using the file uploader.
5.  Click the "Process ID Card" button.
6.  View the original image, processed image, detected text regions, and the extracted information.
7.  Optionally, download the extracted information as a CSV file.

## Potential Improvements

*   Refactor the information extraction logic (`extract_field_info` in `main.py`) to be less reliant on complex regex and positional assumptions. Consider using Named Entity Recognition (NER) models fine-tuned for Vietnamese ID cards for more robust extraction.
*   Improve error handling for edge cases (e.g., very blurry images, unusual lighting).
*   Containerize the application using Docker for easier deployment.
*   Add more comprehensive unit and integration tests.

## 🌟 New Features

### API Endpoints
- **Batch Processing**: Process multiple ID card images in a single request
- **History Management**: View and search processing history with advanced filtering
- **Duplicate Detection**: Automatic detection of previously processed ID cards
- **Auto Port Selection**: Automatic port selection when default ports are in use

### API Documentation

#### 1. Process Single ID Card
```http
POST /process-id-card/
```
Process a single Vietnamese ID card image.

#### 2. Batch Processing
```http
POST /process-batch/
```
Process multiple ID card images in one request.
- Supports up to 10 images per batch
- Automatic duplicate detection
- Configurable processing parameters

#### 3. View Processing History
```http
GET /history
```
Get all processing history with pagination and filtering options:
- Filter by date range
- Filter by ID number
- Filter by success status
- Pagination support (1-100 items per page)

#### 4. Search by ID Number
```http
GET /search/{id_number}
```
Search for specific ID card processing results by ID number.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- MongoDB
- CUDA-capable GPU (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Vietnamese_ID_Card_OCR.git
cd Vietnamese_ID_Card_OCR
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export GEMINI_API_KEY="your_api_key_here"
```

### Running the Application

1. Start the API server:
```bash
python -m src.api.fastapi_app
```

2. Access the API documentation:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## 📝 API Usage Examples

### Process Single ID Card
```python
import requests

url = "http://localhost:8080/process-id-card/"
files = {"file": open("id_card.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

### Batch Processing
```python
import requests

url = "http://localhost:8080/process-batch/"
files = [
    ("files", open("id_card1.jpg", "rb")),
    ("files", open("id_card2.jpg", "rb"))
]
response = requests.post(url, files=files)
print(response.json())
```

### View History
```python
import requests

url = "http://localhost:8080/history"
params = {
    "page": 1,
    "page_size": 10,
    "filter": {
        "start_date": "2024-03-01T00:00:00Z",
        "end_date": "2024-03-20T23:59:59Z",
        "success_only": True
    }
}
response = requests.get(url, params=params)
print(response.json())
```

## 🔧 Configuration

The API supports various configuration options:

### Processing Configuration
- Confidence threshold
- NMS threshold
- Image enhancement
- Processing method selection

### Batch Processing Settings
- Maximum batch size
- Parallel processing option

## 📊 Monitoring

The API includes built-in monitoring features:
- Prometheus metrics endpoint
- Processing time tracking
- Success/error rate monitoring
- Request count tracking

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- YOLO for object detection
- VietOCR for Vietnamese text recognition
- Google Gemini for AI-powered information extraction

