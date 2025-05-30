# Vietnamese ID Card OCR - Refactored & Organized 🚀

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-brightgreen.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive OCR (Optical Character Recognition) system for Vietnamese ID cards using deep learning models, now with a **fully organized, modular, production-ready architecture**.

## ✨ What's New - Complete Reorganization

The project has been completely restructured from a monolithic design into a clean, professional structure following industry best practices:

### 📁 Organized Project Structure

```
VnId-Card/
├── 📄 Core Files
│   ├── README.md                    # This file
│   ├── requirements_windows.txt     # Python dependencies
│   ├── setup.py                     # Package installation
│   ├── Makefile                     # Development automation
│   └── app.py → api_app.py → streamlit_app.py  # Entry points
│
├── 📂 Source Code (src/)
│   ├── api/          # REST API (FastAPI)
│   ├── core/         # Business logic
│   ├── models/       # AI model management
│   ├── utils/        # Utilities
│   ├── ui/           # User interfaces
│   ├── database/     # Database operations
│   └── webhooks/     # Monitoring webhooks
│
├── 📂 Configuration (config/)
│   ├── settings.py   # Centralized configuration
│   └── .env.example  # Environment template
│
├── 📂 Data (data/)
│   ├── models/       # AI model files
│   ├── dictionary/   # Vietnamese dictionaries
│   └── samples/      # Test images
│
├── 📂 Deployment (deployment/)
│   ├── docker/       # Docker configurations
│   ├── k8s/          # Kubernetes manifests
│   └── k3d/          # K3D configurations
│
├── 📂 Scripts (scripts/)
│   ├── setup/        # Installation scripts
│   └── dev/          # Development tools
│
├── 📂 Monitoring (monitoring/)
│   ├── prometheus/   # Metrics collection
│   ├── grafana/      # Dashboards
│   ├── loki/         # Log aggregation
│   └── alertmanager/ # Alert management
│
├── 📂 Documentation (docs/)
│   └── PROJECT_STRUCTURE.md  # Detailed structure guide
│
├── 📂 Tests (tests/)          # Test framework
├── 📄 TEST_FRAMEWORK_STATUS.md # Test coverage and status report
└── 📂 Archive (archive/)      # Legacy files
```

### 🎯 Key Improvements

1. **🏗️ Professional Structure**: Industry-standard project organization
2. **⚙️ Centralized Configuration**: Environment-based settings management
3. **🐳 Production Deployment**: Docker, K8s, and K3D ready
4. **📊 Complete Monitoring**: Prometheus, Grafana, Loki integration
5. **🧪 Complete Testing Framework**: 68 tests implemented and passing
6. **📚 Documentation**: Comprehensive guides and examples
7. **🔧 Developer Tools**: Automated setup and development scripts
8. **🚀 Multiple Deployment Options**: Local, Docker, Kubernetes

### 📁 New Project Structure

```
VnId-Card/
├── src/                        # Main source code
│   ├── models/                 # Model management
│   │   ├── __init__.py
│   │   └── model_manager.py    # Centralized model loading
│   ├── utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── image_processing.py # Image processing utilities
│   │   └── text_processing.py  # Vietnamese text processing
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   └── id_card_processor.py # Main OCR pipeline
│   ├── api/                    # FastAPI application
│   │   ├── __init__.py
│   │   └── fastapi_app.py      # REST API server
│   ├── ui/                     # User interfaces
│   │   ├── __init__.py
│   │   └── streamlit_app.py    # Web interface
│   └── config.py               # Configuration management
├── streamlit_app.py            # Main Streamlit entry point
├── api_app.py                  # Main FastAPI entry point
├── app.py                      # Legacy entry point (now imports new structure)
├── setup.py                    # Package installation
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Multi-service setup
├── Makefile                    # Development commands
├── .env.example                # Environment template
└── requirements_windows.txt    # Dependencies (updated)
```

### ✨ Key Improvements

1. **🏗️ Modular Architecture**: Separated concerns into logical modules
2. **📦 Proper Packaging**: Installable Python package with `setup.py`
3. **🐳 Docker Support**: Containerized deployment ready
4. **⚙️ Configuration Management**: Environment-based configuration
5. **🔌 API-First Design**: Both Streamlit UI and FastAPI REST API
6. **📊 Monitoring**: Built-in Prometheus metrics
7. **🧪 Comprehensive Testing**: 68 unit and integration tests with 98.5% success rate
8. **🚀 Production Ready**: Proper error handling and logging

## 🛠️ Technology Stack

**Core Technologies:**
- **Python 3.9+**: Modern Python with type hints
- **FastAPI**: High-performance REST API framework
- **Streamlit**: Interactive web interface
- **Docker**: Containerized deployment

**AI/ML Libraries:**
- **OpenCV**: Image processing and computer vision
- **PyTorch**: Deep learning framework
- **Ultralytics YOLO**: Object detection models
- **PaddleOCR**: Text detection and recognition
- **VietOCR**: Vietnamese text recognition
- **Google Generative AI**: Advanced text processing

**Additional Tools:**
- **Prometheus**: Metrics and monitoring
- **QReader**: QR code detection
- **Levenshtein**: Text similarity and correction

## 🚀 Quick Start

### Option 1: Using Make (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd VnId-Card

# Install dependencies and setup
make install

# Run Streamlit web interface
make run-streamlit

# Or run FastAPI server
make run-api
```

### Option 2: Manual Installation

```bash
# Create virtual environment
python -m venv venv
venv\\Scripts\\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements_windows.txt
pip install -e .

# Run applications
streamlit run streamlit_app.py
# OR
python api_app.py
```

### Option 3: Docker

```bash
# Build and run with Docker Compose
make docker-run

# Or manually
docker-compose up -d
```

## 📖 Usage

### Web Interface (Streamlit)

1. Start the Streamlit app: `make run-streamlit`
2. Open your browser to `http://localhost:8501`
3. Upload a Vietnamese ID card image
4. Configure processing options in the sidebar
5. Click "Process ID Card" and view results

### REST API (FastAPI)

1. Start the FastAPI server: `make run-api`
2. API documentation available at `http://localhost:8000/docs`
3. Health check: `GET http://localhost:8000/health`
4. Process ID card: `POST http://localhost:8000/process-id-card`

#### Example API Usage:

```python
import requests

# URL of the FastAPI endpoint
url = "http://localhost:8080/process-id-card/"

# Path to the image you want to send
file_path = r"C:\path\to\file\image.jpeg"  

# Open the file and send it via POST request
with open(file_path, "rb") as f:
    files = {"file": (file_path, f, "image/jpeg")}
    headers = {"accept": "application/json"}
    response = requests.post(url, files=files, headers=headers)

# Output the response
print("Status code:", response.status_code)
try:
    print("Response JSON:", response.json())
except Exception as e:
    print("Failed to parse JSON:", str(e))
    print("Raw response:", response.text)
```

```bash
>curl -X POST http://localhost:8080/process-id-card/ \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@C:\path\to\file\image.jpeg"
```

## ⚙️ Configuration

Create a `.env` file based on `.env.example`:

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
# Add your Google AI API key, model paths, etc.
```

Key configuration options:
- `GOOGLE_AI_API_KEY`: For Gemini AI integration
- `MODEL_DIR`: Directory containing AI models
- `LOG_LEVEL`: Logging verbosity
- `ENABLE_GEMINI`: Toggle AI-powered text processing

## 🔧 Development

### Available Commands

```bash
make help              # Show all available commands
make install           # Install package and dependencies
make install-dev       # Install with development dependencies
make test              # Run tests
make lint              # Run code linting
make format            # Format code with black and isort
make clean             # Clean build artifacts
```

### Testing

The project includes a comprehensive test framework with **68 tests** covering all major components:

```bash
# Run all tests
make test
# OR
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_config_*.py -v     # Configuration tests (16 tests)
python -m pytest tests/test_image_processing.py -v  # Image processing tests (18 tests)
python -m pytest tests/test_text_processing_working.py -v  # Text processing tests (10 tests)
python -m pytest tests/test_model_manager.py -v     # Model manager tests (16 tests)
python -m pytest tests/test_api.py -v               # API tests (8 tests + 1 skipped)
```

**Test Coverage:**

- ✅ **Configuration Management**: 16 tests (environment variables, validation, paths)
- ✅ **Image Processing**: 18 tests (resize, enhance, OCR preprocessing, edge cases)
- ✅ **Text Processing**: 10 tests (Vietnamese text handling, extraction, validation)
- ✅ **Model Management**: 16 tests (loading, reloading, device selection, error handling)
- ✅ **API Endpoints**: 8 tests (health checks, metrics, processing endpoints)
- ⏭️ **Integration Test**: 1 skipped (requires real model files for full validation)

**Success Rate**: 98.5% (68 passed, 1 skipped, 0 failed)

For detailed test documentation, see [`TEST_FRAMEWORK_STATUS.md`](./TEST_FRAMEWORK_STATUS.md).

### Adding New Features

1. **Models**: Add to `src/models/`
2. **Utilities**: Add to `src/utils/`
3. **Core Logic**: Extend `src/core/`
4. **API Endpoints**: Extend `src/api/`
5. **UI Components**: Extend `src/ui/`

## 📊 Monitoring

The FastAPI application includes built-in Prometheus metrics:

- Request counters
- Response time histograms
- Error rates
- Model performance metrics

Access metrics at: `http://localhost:8000/metrics`

## 🐳 Docker Deployment

### Development

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Production

```bash
# Build production image
docker build -t vnid-card-ocr .

# Run with custom configuration
docker run -p 8000:8000 -e GOOGLE_AI_API_KEY=your_key vnid-card-ocr
```

## 🔒 Security Notes

- Store API keys in environment variables, not in code
- Use HTTPS in production
- Implement proper authentication for API endpoints
- Validate and sanitize all inputs

## 🤝 Migration from Original Code

If you're migrating from the original monolithic `app.py`:

1. **Models**: Now managed by `ModelManager` class
2. **Text Processing**: Moved to `src/utils/text_processing.py`
3. **Image Processing**: Moved to `src/utils/image_processing.py`
4. **OCR Pipeline**: Centralized in `IDCardProcessor`
5. **Configuration**: Environment-based with `Config` class

The original code is preserved in `app_backup.py` for reference.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- VietOCR team for Vietnamese text recognition
- Ultralytics for YOLO models
- PaddleOCR for text detection
- The Vietnamese OCR community

---

**Note**: This refactored version maintains full compatibility with the original functionality while providing a much more maintainable and scalable codebase.
