# Vietnamese ID Card OCR - Complete Project Guide 🚀

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-brightgreen.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![Kubernetes](https://img.shields.io/badge/K8s-Ready-purple.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

Hướng dẫn hoàn chỉnh cho hệ thống Vietnamese ID Card OCR - một giải pháp OCR hiện đại, có cấu trúc chuyên nghiệp và sẵn sàng cho production với khả năng triển khai đa nền tảng.

---

## 📋 Mục lục

- [1. Tổng quan dự án](#1-tổng-quan-dự-án)
- [2. Cấu trúc dự án](#2-cấu-trúc-dự-án)
- [3. Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
- [4. Cài đặt và thiết lập](#4-cài-đặt-và-thiết-lập)
- [5. Cách sử dụng](#5-cách-sử-dụng)
- [6. API Documentation](#6-api-documentation)
- [7. Deployment](#7-deployment)
- [8. Monitoring & Logging](#8-monitoring--logging)
- [9. Testing](#9-testing)
- [10. Configuration](#10-configuration)
- [11. Troubleshooting](#11-troubleshooting)
- [12. Development Guide](#12-development-guide)
- [13. Security](#13-security)
- [14. Performance](#14-performance)
- [15. Migration Guide](#15-migration-guide)

---

## 1. Tổng quan dự án

### 🎯 Mục tiêu
Vietnamese ID Card OCR là một hệ thống OCR (Optical Character Recognition) toàn diện cho việc nhận dạng và trích xuất thông tin từ thẻ căn cước công dân Việt Nam sử dụng công nghệ deep learning hiện đại.

### ✨ Tính năng chính
- **🔍 OCR thông minh**: Trích xuất thông tin từ CCCD với độ chính xác cao
- **🌐 Multi-Interface**: Hỗ trợ cả Web UI (Streamlit) và REST API (FastAPI)
- **🧠 AI Integration**: Tích hợp Google Generative AI và VietOCR
- **📊 Monitoring**: Hệ thống giám sát toàn diện với Prometheus, Grafana, Loki
- **🐳 Container Ready**: Hỗ trợ Docker và Kubernetes deployment
- **🧪 Comprehensive Testing**: 68 unit tests với tỷ lệ thành công 98.5%
- **🔧 Production Ready**: Cấu hình environment-based, error handling chuyên nghiệp

### 🏗️ Kiến trúc
- **Modular Design**: Tách biệt rõ ràng các thành phần
- **Microservices Ready**: Có thể triển khai độc lập từng service
- **Scalable**: Thiết kế để mở rộng theo chiều ngang
- **Cloud Native**: Tối ưu cho deployment trên cloud

---

## 2. Cấu trúc dự án

```
VnId-Card/
├── 📄 README.md                        # Tài liệu chính
├── 📄 requirements.txt                  # Dependencies Python
├── 📄 setup.py                         # Package configuration
├── 📄 Makefile                         # Automation commands
├── 📄 .env                             # Environment variables
├── 📄 api_app.py                       # FastAPI entry point
├── 📄 streamlit_app.py                 # Streamlit entry point
├── 📄 app.py                           # Legacy compatibility
│
├── 📂 src/                             # Source code chính
│   ├── 📂 api/                         # REST API implementation
│   │   └── 📄 fastapi_app.py           # FastAPI application
│   ├── 📂 core/                        # Business logic chính
│   │   └── 📄 id_card_processor.py     # OCR processing pipeline
│   ├── 📂 models/                      # AI model management
│   │   └── 📄 model_manager.py         # Model loading & management
│   ├── 📂 utils/                       # Utility functions
│   │   ├── 📄 image_processing.py      # Image processing
│   │   └── 📄 text_processing.py       # Vietnamese text processing
│   ├── 📂 ui/                          # User interfaces
│   │   └── 📄 streamlit_app.py         # Streamlit web interface
│   ├── 📂 database/                    # Database operations
│   │   ├── 📄 models.py                # Database models
│   │   └── 📄 mongodb.py               # MongoDB integration
│   └── 📂 webhooks/                    # Webhook handlers
│
├── 📂 config/                          # Configuration files
│   ├── 📄 settings.py                  # Configuration management
│   └── 📄 .env.example                 # Environment template
│
├── 📂 data/                            # Data files
│   ├── 📂 models/                      # AI model files
│   │   ├── 📂 corner_detection_model/  # YOLO corner detection
│   │   └── 📂 yolo_detect_text/        # YOLO text detection
│   ├── 📂 dictionary/                  # Vietnamese dictionaries
│   ├── 📂 samples/                     # Sample images
│   ├── 📂 outputs/                     # Processing outputs
│   └── 📂 uploads/                     # Upload directory
│
├── 📂 deployment/                      # Deployment configurations
│   ├── 📂 docker/                      # Docker configurations
│   │   ├── 📄 docker-compose.yml       # Docker Compose main
│   │   ├── 📄 docker-compose.k3d.yml   # K3D deployment
│   │   └── 📄 Dockerfile               # Container image
│   ├── 📂 k8s/                         # Kubernetes manifests
│   │   ├── 📄 deployment.yaml          # K8s deployment
│   │   ├── 📄 service.yaml             # K8s service
│   │   ├── 📄 ingress.yaml             # K8s ingress
│   │   └── 📄 autoscaling.yaml         # HPA configuration
│   └── 📂 k3d/                         # K3D configurations
│       └── 📄 k3d-config.yaml          # K3D cluster config
│
├── 📂 monitor/                         # Monitoring & logging
│   ├── 📄 start-monitoring.bat         # Start monitoring (Windows)
│   ├── 📄 start-monitoring.sh          # Start monitoring (Linux)
│   ├── 📂 prometheus/                  # Prometheus configuration
│   │   ├── 📄 prometheus.yml           # Prometheus config
│   │   └── 📄 alert-rules.yml          # Alert rules
│   ├── 📂 grafana/                     # Grafana dashboards
│   │   ├── 📂 dashboards/              # Dashboard definitions
│   │   └── 📂 provisioning/            # Auto-provisioning
│   ├── 📂 alertmanager/                # Alert management
│   │   └── 📄 alertmanager.yml         # Alert routing
│   ├── 📂 loki/                        # Log aggregation
│   │   └── 📄 loki-config.yml          # Loki configuration
│   └── 📂 fluent-bit/                  # Log collection
│       ├── 📄 fluent-bit.conf          # Fluent Bit config
│       └── 📄 parsers.conf             # Log parsers
│
├── 📂 scripts/                         # Utility scripts
│   ├── 📂 setup/                       # Setup scripts
│   │   ├── 📄 check-prerequisites.ps1  # Windows prerequisites
│   │   └── 📄 quick-start.ps1          # Quick start script
│   └── 📂 dev/                         # Development tools
│       ├── 📄 deploy-k3d.ps1           # K3D deployment
│       └── 📄 test-deployment.ps1      # Deployment testing
│
├── 📂 tests/                           # Test framework
│   ├── 📄 conftest.py                  # Test configuration
│   ├── 📄 run_tests.py                 # Test runner
│   ├── 📄 test_config_*.py             # Configuration tests (16 tests)
│   ├── 📄 test_image_processing.py     # Image processing tests (18 tests)
│   ├── 📄 test_text_processing_*.py    # Text processing tests (10 tests)
│   ├── 📄 test_model_manager_*.py      # Model management tests (16 tests)
│   ├── 📄 test_api_*.py                # API tests (8 tests)
│   └── 📂 data/                        # Test data
│
├── 📂 docs/                            # Documentation
│   ├── 📄 DEPLOYMENT_GUIDE.md          # Deployment guide
│   ├── 📄 COMPREHENSIVE_DEPLOYMENT_GUIDE.md # Complete deployment guide
│   ├── 📄 K3D-DEPLOYMENT.md            # K3D deployment guide
│   ├── 📄 MONITORING_README.md         # Monitoring documentation
│   └── 📄 PROJECT_STRUCTURE.md         # Project structure guide
│
└── 📂 logs/                            # Log files
    ├── 📄 api.log                      # API logs
    ├── 📄 error.log                    # Error logs
    ├── 📄 metrics.log                  # Metrics logs
    └── 📄 model.log                    # Model logs
```

---

## 3. Yêu cầu hệ thống

### 💻 System Requirements

#### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.9+
- **RAM**: 4GB (khuyến nghị 8GB+)
- **Storage**: 10GB free space
- **Network**: Kết nối internet ổn định

#### Development Requirements
- **Python**: 3.9+ với pip
- **Git**: Latest version
- **Code Editor**: VS Code, PyCharm, hoặc tương tự

#### Production Requirements
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **Kubernetes**: 1.25+ (optional)
- **Load Balancer**: Nginx, HAProxy (production)

### 🛠️ Technology Stack

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

**Monitoring & Observability:**
- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Loki**: Log aggregation and storage
- **Alertmanager**: Alert routing and notifications
- **Fluent Bit**: Log collection and processing

**Additional Tools:**
- **QReader**: QR code detection
- **Levenshtein**: Text similarity and correction
- **MongoDB**: Database for results storage

---

## 4. Cài đặt và thiết lập

### 🚀 Quick Start (5 phút)

#### Phương pháp 1: Automated Setup (Khuyến nghị)

**Windows:**
```powershell
# Clone repository
git clone https://github.com/your-repo/VnId-Card.git
cd VnId-Card

# Run automated setup
.\scripts\setup\quick-start.ps1
```

**Linux/macOS:**
```bash
# Clone repository
git clone https://github.com/your-repo/VnId-Card.git
cd VnId-Card

# Run automated setup
chmod +x scripts/setup/quick-start.sh
./scripts/setup/quick-start.sh
```

#### Phương pháp 2: Manual Setup

**Bước 1: Cài đặt Dependencies**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

**Bước 2: Configuration**
```bash
# Copy environment template
cp config/.env.example .env

# Edit configuration
# Windows:
notepad .env
# Linux/macOS:
nano .env
```

**Bước 3: Download Models**
```bash
# Download required models (script will be provided)
python scripts/download_models.py
```

### ⚙️ Environment Configuration

Tạo file `.env` trong thư mục root:

```env
# API Configuration
GOOGLE_AI_API_KEY=your_google_ai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# Application Settings
LOG_LEVEL=INFO
DEBUG=false
UPLOAD_FOLDER=data/uploads
OUTPUT_FOLDER=data/outputs

# Model Paths
YOLO_TEXT_MODEL_PATH=data/models/yolo_detect_text/best.pt
YOLO_TEXT_MODEL_V2_PATH=data/models/yolo_detect_text/bestv2.pt
CORNER_DETECTION_MODEL_PATH=data/models/corner_detection_model/weight/29_03_25-YOLOv11n-Corner-best_metrics.pt

# Server Configuration
STREAMLIT_PORT=8501
FASTAPI_PORT=8000
FASTAPI_HOST=0.0.0.0

# Database Configuration (Optional)
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=vnid_card_ocr

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
LOKI_PORT=3100
ALERTMANAGER_PORT=9093

# Alert Notification Channels (Optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/your/slack/webhook
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Email Configuration (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### 🔧 Available Commands

```bash
# Show all available commands
make help

# Setup and Installation
make install           # Install package and dependencies
make install-dev       # Install with development dependencies
make setup             # Setup environment and validate models

# Running Applications
make run-streamlit     # Run Streamlit web interface
make run-api           # Run FastAPI server

# Development
make test              # Run tests
make lint              # Run linting checks
make format            # Format code
make clean             # Clean build artifacts

# Docker
make docker-build      # Build Docker image
make docker-run        # Run with Docker Compose

# Deployment
make deploy-k3d        # Deploy to K3D cluster
make deploy-k8s        # Deploy to Kubernetes

# Monitoring
make start-monitoring  # Start monitoring stack
make stop-monitoring   # Stop monitoring stack
```

---

## 5. Cách sử dụng

### 🌐 Web Interface (Streamlit)

```bash
# Start Streamlit application
make run-streamlit
# hoặc
streamlit run streamlit_app.py
```

**Truy cập:** http://localhost:8501

**Tính năng:**
- Upload ảnh CCCD
- Xem kết quả OCR real-time
- Điều chỉnh cấu hình processing
- Download kết quả JSON/CSV
- Xem confidence scores

### 🔗 REST API (FastAPI)

```bash
# Start FastAPI server
make run-api
# hoặc
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

**Truy cập:** http://localhost:8000

**API Documentation:** http://localhost:8000/docs

#### Example API Usage:

**Health Check:**
```bash
curl -X GET "http://localhost:8000/health"
```

**Process ID Card:**
```bash
curl -X POST "http://localhost:8000/process-id-card" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/id_card.jpg"
```

**Get Metrics:**
```bash
curl -X GET "http://localhost:8000/metrics"
```

---

## 6. API Documentation

### 📋 Endpoints Overview

| Method | Endpoint | Description | Parameters |
|--------|----------|-------------|------------|
| GET | `/health` | Health check | None |
| GET | `/` | Root endpoint | None |
| POST | `/process-id-card` | Process ID card image | `file` (multipart) |
| GET | `/metrics` | Prometheus metrics | None |
| POST | `/webhooks/test-alert` | Test alert webhook | `alert_type`, `severity` |

### 📤 Request/Response Examples

#### Process ID Card

**Request:**
```bash
curl -X POST "http://localhost:8000/process-id-card" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@id_card.jpg"
```

**Response:**
```json
{
  "status": "success",
  "method": "vietocr",
  "processing_time": 2.45,
  "extracted_info": {
    "ID_number": "123456789012",
    "Name": "Nguyen Van A",
    "Date_of_birth": "01/01/1990",
    "Gender": "Nam",
    "Nationality": "Việt Nam",
    "Place_of_origin": "Hà Nội, Việt Nam",
    "Place_of_residence": "123 Đường ABC, Quận 1, TP.HCM"
  },
  "confidence_scores": {
    "overall": 0.95,
    "fields": {
      "ID_number": 0.98,
      "Name": 0.92,
      "Date_of_birth": 0.94
    }
  },
  "detected_regions": [
    {
      "field": "ID_number",
      "bbox": [10, 20, 200, 50],
      "confidence": 0.98
    }
  ]
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Invalid image format",
  "error_code": "INVALID_FORMAT",
  "processing_time": 0.1
}
```

---

## 7. Deployment

### 🐳 Docker Deployment

#### Development Environment

```bash
# Build và run với Docker Compose
docker-compose up -d

# Xem logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Production Environment

```bash
# Build production image
docker build -t vnid-card-ocr:latest .

# Run với production configuration
docker run -d \
  --name vnid-card-ocr \
  -p 8000:8000 \
  -p 8501:8501 \
  -e GOOGLE_AI_API_KEY=your_key \
  -e LOG_LEVEL=INFO \
  -v $(pwd)/data:/app/data \
  vnid-card-ocr:latest
```

### ☸️ Kubernetes Deployment

#### Chuẩn bị

```bash
# Apply namespace và RBAC
kubectl apply -f deployment/k8s/namespace-rbac.yaml

# Create ConfigMap và Secrets
kubectl create configmap vnid-config --from-env-file=.env
kubectl create secret generic vnid-secrets --from-env-file=.env
```

#### Deploy Application

```bash
# Deploy persistent volumes
kubectl apply -f deployment/k8s/persistent-volumes.yaml

# Deploy application
kubectl apply -f deployment/k8s/deployment.yaml

# Deploy service
kubectl apply -f deployment/k8s/service.yaml

# Deploy ingress
kubectl apply -f deployment/k8s/ingress.yaml

# Deploy autoscaling
kubectl apply -f deployment/k8s/autoscaling.yaml
```

#### Verify Deployment

```bash
# Check pods
kubectl get pods -l app=vnidcard-app

# Check services
kubectl get svc

# View logs
kubectl logs -l app=vnidcard-app -f

# Port forward for testing
kubectl port-forward svc/vnidcard-service 8000:8000
```

### 🚀 K3D Local Kubernetes

#### Quick K3D Deployment

```bash
# Run automated K3D deployment
.\scripts\dev\deploy-k3d.ps1

# Hoặc sử dụng Makefile
make deploy-k3d
```

#### Manual K3D Setup

```bash
# Create K3D cluster
k3d cluster create vnidcard-cluster \
  --agents 2 \
  --registry-create vnidcard-registry:5000 \
  --port "8501:8501@loadbalancer" \
  --port "8080:8080@loadbalancer"

# Build và push image
docker build -t vnidcard-app:latest .
docker tag vnidcard-app:latest localhost:5000/vnidcard-app:latest
docker push localhost:5000/vnidcard-app:latest

# Deploy to K3D
kubectl apply -f deployment/k8s/
```

#### Access Applications

- **Streamlit UI**: http://localhost:8501
- **API Endpoint**: http://localhost:8080
- **Registry**: http://localhost:5000

### 🌟 Production Best Practices

#### High Availability Setup

```yaml
# deployment/k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vnidcard-deployment
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
  selector:
    matchLabels:
      app: vnidcard-app
  template:
    spec:
      containers:
      - name: vnidcard-container
        image: vnidcard-app:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

#### Load Balancer Configuration

```yaml
# deployment/k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vnidcard-service
spec:
  type: LoadBalancer
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: ui
    port: 8501
    targetPort: 8501
  selector:
    app: vnidcard-app
```

---

## 8. Monitoring & Logging

### 📊 Monitoring Stack

Hệ thống monitoring hoàn chỉnh bao gồm:

- **Prometheus**: Metrics collection (Port 9090)
- **Grafana**: Visualization và dashboards (Port 3000)
- **Loki**: Log aggregation (Port 3100)
- **Alertmanager**: Alert routing (Port 9093)
- **Fluent Bit**: Log collection
- **Node Exporter**: System metrics (Port 9100)
- **cAdvisor**: Container metrics (Port 8080)

#### Start Monitoring Stack

**Windows:**
```powershell
cd monitor
.\start-monitoring.bat
```

**Linux/macOS:**
```bash
cd monitor
chmod +x start-monitoring.sh
./start-monitoring.sh
```

#### Access Monitoring Services

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Alertmanager**: http://localhost:9093
- **API Metrics**: http://localhost:8000/metrics

### 📈 Dashboards

#### 1. API Monitoring Dashboard
- Request rate và response time
- Error rates và status codes
- Confidence score distributions
- Processing time metrics

#### 2. System Monitoring Dashboard
- CPU, Memory, Disk usage
- Network I/O
- Container metrics
- GPU utilization (nếu có)

#### 3. Logs Dashboard
- Real-time log streaming
- Error log filtering
- Search và query capabilities
- Log level distributions

### 🚨 Alert Rules

#### High Priority Alerts

| Alert | Condition | Severity | Action |
|-------|-----------|----------|---------|
| APIHighErrorRate | Error rate > 50% for 5min | Critical | All channels |
| APILowConfidence | Confidence < 0.6 for 10min | Warning | Slack + Email |
| SystemHighMemory | Memory > 90% for 15min | Critical | All channels |
| ContainerDown | Container not running | Critical | All channels |

#### Alert Testing

```bash
# Test different alert severities
curl -X POST "http://localhost:8000/webhooks/test-alert?alert_type=test&severity=info"
curl -X POST "http://localhost:8000/webhooks/test-alert?alert_type=performance&severity=warning"  
curl -X POST "http://localhost:8000/webhooks/test-alert?alert_type=system&severity=critical"
```

### 📝 Log Management

#### Log Levels
- **DEBUG**: Detailed tracing information
- **INFO**: General information messages
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical error messages

#### Log Files
- `logs/api.log`: API request/response logs
- `logs/error.log`: Error và exception logs
- `logs/metrics.log`: Performance metrics logs
- `logs/model.log`: Model loading và inference logs

#### Log Rotation
```bash
# Manual log cleanup
.\monitor\cleanup-logs.bat

# With backup
.\monitor\cleanup-logs.bat --backup
```

---

## 9. Testing

### 🧪 Test Framework

Dự án bao gồm 68 tests toàn diện với tỷ lệ thành công 98.5%:

#### Test Categories

**Configuration Tests (16 tests)**
```bash
python -m pytest tests/test_config_*.py -v
```
- Environment variables validation
- Configuration loading
- Path management
- Settings validation

**Image Processing Tests (18 tests)**
```bash
python -m pytest tests/test_image_processing.py -v
```
- Image resize và enhancement
- NMS và IoU calculations
- QR code detection
- Edge case handling

**Text Processing Tests (10 tests)**
```bash
python -m pytest tests/test_text_processing_working.py -v
```
- Vietnamese text extraction
- Gender parsing
- Address components
- OCR artifact cleaning

**Model Management Tests (16 tests)**
```bash
python -m pytest tests/test_model_manager.py -v
```
- Model loading và reloading
- Device selection
- Error handling
- State persistence

**API Tests (8 tests + 1 skipped)**
```bash
python -m pytest tests/test_api.py -v
```
- Health checks
- Metrics endpoints
- File upload processing
- Error responses

#### Running Tests

```bash
# Run all tests
make test
# hoặc
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/test_config_*.py -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run specific test
python -m pytest tests/test_api.py::test_health_endpoint -v
```

#### Test Configuration

File `tests/conftest.py` chứa:
- Test fixtures
- Mock configurations
- Test data setup
- Common test utilities

#### Test Data

Thư mục `tests/data/` chứa:
- Sample ID card images
- Expected output data
- Mock model files
- Test configuration files

---

## 10. Configuration

### ⚙️ Configuration Management

Hệ thống sử dụng environment-based configuration với class `Config` trong `config/settings.py`.

#### Configuration Hierarchy

1. **Environment Variables** (highest priority)
2. **`.env` file** 
3. **Default values** (lowest priority)

#### Core Configuration Sections

```python
# API Configuration
GOOGLE_AI_API_KEY: str
GEMINI_API_KEY: str
LOG_LEVEL: str = "INFO"
DEBUG: bool = False

# Server Configuration  
FASTAPI_HOST: str = "0.0.0.0"
FASTAPI_PORT: int = 8000
STREAMLIT_PORT: int = 8501

# Model Paths
YOLO_TEXT_MODEL_PATH: str
YOLO_TEXT_MODEL_V2_PATH: str
CORNER_DETECTION_MODEL_PATH: str

# Storage Paths
UPLOAD_FOLDER: str = "data/uploads"
OUTPUT_FOLDER: str = "data/outputs"
```

#### Environment-specific Configurations

**Development:**
```env
DEBUG=true
LOG_LEVEL=DEBUG
FASTAPI_HOST=127.0.0.1
```

**Production:**
```env
DEBUG=false
LOG_LEVEL=INFO
FASTAPI_HOST=0.0.0.0
FASTAPI_WORKERS=4
```

**Testing:**
```env
LOG_LEVEL=WARNING
UPLOAD_FOLDER=tests/data/uploads
OUTPUT_FOLDER=tests/data/outputs
```

### 🔧 Advanced Configuration

#### Model Configuration
```env
# Model performance settings
MODEL_DEVICE=auto  # auto, cpu, cuda
MODEL_BATCH_SIZE=1
MODEL_CONFIDENCE_THRESHOLD=0.5
MODEL_NMS_THRESHOLD=0.4

# Processing options
ENABLE_GPU=true
ENABLE_PREPROCESSING=true
ENABLE_POSTPROCESSING=true
```

#### Database Configuration
```env
# MongoDB settings
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=vnid_card_ocr
MONGODB_COLLECTION_OCR_RESULTS=ocr_results
MONGODB_COLLECTION_METRICS=metrics
MONGODB_CONNECTION_TIMEOUT=5000
```

#### Monitoring Configuration
```env
# Prometheus settings
PROMETHEUS_RETENTION=15d
PROMETHEUS_SCRAPE_INTERVAL=15s

# Loki settings
LOKI_RETENTION=7d
LOKI_COMPRESSION=gzip

# Alert settings
ALERT_EVALUATION_INTERVAL=30s
ALERT_NOTIFICATION_TIMEOUT=10s
```

---

## 11. Troubleshooting

### 🔧 Common Issues

#### Issue 1: Model Loading Errors

**Symptoms:**
- "Model file not found" errors
- "CUDA out of memory" errors
- Slow model initialization

**Solutions:**
```bash
# Check model files exist
ls -la data/models/

# Download missing models
python scripts/download_models.py

# Check GPU memory
nvidia-smi

# Force CPU mode if GPU issues
export MODEL_DEVICE=cpu
```

#### Issue 2: API Performance Issues

**Symptoms:**
- Slow response times
- High memory usage
- Connection timeouts

**Solutions:**
```bash
# Check system resources
docker stats

# Increase memory limits
# In docker-compose.yml:
mem_limit: 4g
memswap_limit: 4g

# Enable model caching
export ENABLE_MODEL_CACHING=true
```

#### Issue 3: Monitoring Not Working

**Symptoms:**
- Grafana shows "No data"
- Prometheus targets down
- Missing metrics

**Solutions:**
```bash
# Check Prometheus targets
curl http://localhost:9090/targets

# Verify API metrics endpoint
curl http://localhost:8000/metrics

# Restart monitoring stack
cd monitor
.\stop-monitoring.bat
.\start-monitoring.bat
```

#### Issue 4: Image Processing Errors

**Symptoms:**
- "Invalid image format" errors
- Poor OCR accuracy
- Processing failures

**Solutions:**
```bash
# Check image format support
python -c "from PIL import Image; print(Image.EXTENSION)"

# Validate image file
python -c "from PIL import Image; Image.open('test.jpg').verify()"

# Check image dimensions
python -c "from PIL import Image; print(Image.open('test.jpg').size)"
```

### 🔍 Debugging Commands

#### Application Debugging
```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose logging
python -c "import logging; logging.basicConfig(level=logging.DEBUG)"

# Check configuration
python -c "from config.settings import get_config; print(get_config().__dict__)"
```

#### Docker Debugging
```bash
# Check container logs
docker-compose logs -f vnid-card-api

# Access container shell
docker-compose exec vnid-card-api bash

# Check container resource usage
docker stats

# Inspect container configuration
docker inspect vnid-card-ocr
```

#### Kubernetes Debugging
```bash
# Check pod status
kubectl get pods -l app=vnidcard-app

# View pod logs
kubectl logs -l app=vnidcard-app -f

# Describe pod for events
kubectl describe pod <pod-name>

# Access pod shell
kubectl exec -it <pod-name> -- bash

# Check resource usage
kubectl top pods
```

### 📊 Performance Tuning

#### API Performance
```python
# In src/api/fastapi_app.py
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# Add performance middleware
@app.middleware("http")
async def add_performance_headers(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Run with optimized settings
if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=4,  # Adjust based on CPU cores
        loop="uvloop",  # Better performance on Linux
        http="httptools"  # Faster HTTP parsing
    )
```

#### Model Performance
```python
# In src/models/model_manager.py
import torch

class ModelManager:
    def __init__(self):
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Use mixed precision if GPU available
        self.use_amp = torch.cuda.is_available()
        
    def optimize_model(self, model):
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        # Enable eval mode for inference
        model.eval()
        
        return model
```

---

## 12. Development Guide

### 🛠️ Development Setup

#### Prerequisites
```bash
# Install development tools
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Setup IDE configuration
# For VS Code: install Python, Docker, Kubernetes extensions
```

#### Development Workflow

**1. Code Standards**
```bash
# Format code
make format
# hoặc
black src/ tests/
isort src/ tests/

# Lint code
make lint
# hoặc
flake8 src/ tests/
mypy src/

# Run tests
make test
```

**2. Adding New Features**

**Models**: Add to `src/models/`
```python
# src/models/new_model.py
class NewModel:
    def __init__(self):
        pass
    
    def load_model(self):
        # Implementation
        pass
```

**Utilities**: Add to `src/utils/`
```python
# src/utils/new_utility.py
def new_utility_function():
    # Implementation
    pass
```

**API Endpoints**: Extend `src/api/fastapi_app.py`
```python
@app.post("/new-endpoint")
async def new_endpoint():
    # Implementation
    pass
```

**Tests**: Add corresponding tests
```python
# tests/test_new_feature.py
def test_new_feature():
    # Test implementation
    assert True
```

**3. Database Integration**

```python
# src/database/models.py
from datetime import datetime
from typing import Dict, Any

class NewModel:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.created_at = datetime.utcnow()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "created_at": self.created_at.isoformat()
        }
```

### 🏗️ Architecture Patterns

#### Dependency Injection
```python
# src/core/id_card_processor.py
class IDCardProcessor:
    def __init__(self, model_manager: ModelManager, db_client: Optional[DatabaseClient] = None):
        self.model_manager = model_manager
        self.db_client = db_client or get_db_client()
```

#### Factory Pattern
```python
# src/models/model_factory.py
class ModelFactory:
    @staticmethod
    def create_model(model_type: str):
        if model_type == "yolo":
            return YOLOModel()
        elif model_type == "vietocr":
            return VietOCRModel()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

#### Observer Pattern for Monitoring
```python
# src/monitoring/metrics_collector.py
class MetricsCollector:
    def __init__(self):
        self.observers = []
    
    def add_observer(self, observer):
        self.observers.append(observer)
    
    def notify(self, metric_data):
        for observer in self.observers:
            observer.update(metric_data)
```

### 📋 Code Review Checklist

- [ ] Code follows project structure
- [ ] Tests written và passing
- [ ] Documentation updated
- [ ] Environment variables documented
- [ ] Error handling implemented
- [ ] Logging added appropriately
- [ ] Performance considerations addressed
- [ ] Security implications reviewed

---

## 13. Security

### 🔒 Security Best Practices

#### API Security
```python
# src/api/fastapi_app.py
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Verify JWT token
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@app.post("/secure-endpoint")
async def secure_endpoint(token_data = Depends(verify_token)):
    # Protected endpoint
    pass
```

#### Input Validation
```python
from pydantic import BaseModel, validator
from typing import Optional

class ProcessRequest(BaseModel):
    confidence_threshold: Optional[float] = 0.5
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence threshold must be between 0 and 1')
        return v
```

#### File Upload Security
```python
# src/utils/file_validation.py
import magic
from pathlib import Path

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_image_file(file_path: str) -> bool:
    # Check file extension
    if Path(file_path).suffix.lower() not in ALLOWED_EXTENSIONS:
        return False
    
    # Check file size
    if Path(file_path).stat().st_size > MAX_FILE_SIZE:
        return False
    
    # Check MIME type
    mime = magic.from_file(file_path, mime=True)
    if not mime.startswith('image/'):
        return False
    
    return True
```

#### Environment Variables Security
```bash
# Use strong, unique API keys
GOOGLE_AI_API_KEY=your_secure_api_key_here

# Restrict API access
ALLOWED_ORIGINS=["https://yourdomain.com"]
CORS_ENABLED=false

# Enable HTTPS in production
HTTPS_ONLY=true
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

#### Container Security
```dockerfile
# Dockerfile
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Change ownership to appuser
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 🛡️ Security Monitoring

#### Security Metrics
```python
# src/monitoring/security_metrics.py
from prometheus_client import Counter, Histogram

# Security metrics
failed_auth_attempts = Counter('failed_auth_attempts_total', 'Failed authentication attempts')
suspicious_requests = Counter('suspicious_requests_total', 'Suspicious requests detected')
file_upload_size = Histogram('file_upload_size_bytes', 'File upload sizes')

def track_failed_auth():
    failed_auth_attempts.inc()

def track_suspicious_request():
    suspicious_requests.inc()
```

#### Security Alerts
```yaml
# monitor/prometheus/alert-rules.yml
groups:
  - name: security.rules
    rules:
      - alert: HighFailedAuthRate
        expr: rate(failed_auth_attempts_total[5m]) > 10
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High failed authentication rate detected"
          
      - alert: SuspiciousActivity
        expr: rate(suspicious_requests_total[5m]) > 5
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Suspicious activity detected"
```

---

## 14. Performance

### ⚡ Performance Optimization

#### API Performance
```python
# src/api/fastapi_app.py
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Optimize CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=3600,  # Cache preflight requests
)

# Connection pooling for external services
@app.on_event("startup")
async def startup_event():
    app.state.http_client = httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
    )

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.http_client.aclose()
```

#### Model Performance
```python
# src/models/model_manager.py
import torch
from functools import lru_cache

class ModelManager:
    def __init__(self):
        self.device = self._get_optimal_device()
        self.models = {}
        
    def _get_optimal_device(self):
        if torch.cuda.is_available():
            # Use the GPU with most memory
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                gpu_memory = [torch.cuda.get_device_properties(i).total_memory 
                             for i in range(gpu_count)]
                return f"cuda:{gpu_memory.index(max(gpu_memory))}"
            return "cuda:0"
        return "cpu"
    
    @lru_cache(maxsize=3)
    def load_model(self, model_path: str):
        # Cache models in memory
        if model_path not in self.models:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            
            # Optimize for inference
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead")
                
            self.models[model_path] = model
            
        return self.models[model_path]
```

#### Image Processing Optimization
```python
# src/utils/image_processing.py
import cv2
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def optimize_image_processing(image: np.ndarray) -> np.ndarray:
    # Use OpenCV for faster processing
    if len(image.shape) == 3:
        # Convert to RGB if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Optimize image size for processing
    height, width = image.shape[:2]
    if width > 1920 or height > 1080:
        scale = min(1920/width, 1080/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

def parallel_image_processing(images: list) -> list:
    """Process multiple images in parallel"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(optimize_image_processing, images))
    return results
```

### 📊 Performance Monitoring

#### Custom Metrics
```python
# src/monitoring/performance_metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time
import functools

# Performance metrics
request_duration = Histogram('http_request_duration_seconds', 
                            'Request duration in seconds', ['method', 'endpoint'])
model_inference_time = Histogram('model_inference_duration_seconds',
                                'Model inference time in seconds', ['model_type'])
memory_usage = Gauge('memory_usage_bytes', 'Memory usage in bytes')
gpu_utilization = Gauge('gpu_utilization_percent', 'GPU utilization percentage')

def track_performance(metric_name: str):
    """Decorator to track function performance"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                if metric_name == 'model_inference':
                    model_inference_time.labels(model_type=func.__name__).observe(duration)
                elif metric_name == 'api_request':
                    request_duration.labels(method='POST', endpoint=func.__name__).observe(duration)
        return wrapper
    return decorator
```

#### Load Testing
```python
# scripts/load_test.py
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor

async def make_request(session, url, data):
    async with session.post(url, data=data) as response:
        return await response.json()

async def load_test(concurrent_requests=10, total_requests=100):
    url = "http://localhost:8000/process-id-card"
    
    # Prepare test data
    with open("tests/data/sample_id_card.jpg", "rb") as f:
        test_data = {"file": f.read()}
    
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def bounded_request():
            async with semaphore:
                return await make_request(session, url, test_data)
        
        # Execute requests
        tasks = [bounded_request() for _ in range(total_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Calculate statistics
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        failed_requests = total_requests - successful_requests
        total_time = end_time - start_time
        requests_per_second = total_requests / total_time
        
        print(f"Load Test Results:")
        print(f"Total Requests: {total_requests}")
        print(f"Successful: {successful_requests}")
        print(f"Failed: {failed_requests}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Requests/Second: {requests_per_second:.2f}")

if __name__ == "__main__":
    asyncio.run(load_test())
```

---

## 15. Migration Guide

### 🔄 Migration from Original Code

Nếu bạn đang migrate từ code monolithic `app.py` gốc:

#### Key Changes

| Original | New Location | Description |
|----------|--------------|-------------|
| `app.py` | `src/core/id_card_processor.py` | Main processing logic |
| Model loading | `src/models/model_manager.py` | Centralized model management |
| Image utilities | `src/utils/image_processing.py` | Image processing functions |
| Text utilities | `src/utils/text_processing.py` | Vietnamese text processing |
| Streamlit UI | `src/ui/streamlit_app.py` | Web interface |
| API endpoints | `src/api/fastapi_app.py` | REST API |

#### Migration Steps

**1. Backup Original Code**
```bash
cp app.py archive/app_backup.py
cp -r original_structure/ archive/
```

**2. Update Import Statements**
```python
# Old imports
from app import process_image

# New imports  
from src.core.id_card_processor import IDCardProcessor
```

**3. Update Configuration**
```python
# Old configuration
config = {
    "api_key": "your_key",
    "model_path": "models/best.pt"
}

# New configuration (use environment variables)
from config.settings import get_config
config = get_config()
```

**4. Update Function Calls**
```python
# Old way
result = process_image(image_path)

# New way
processor = IDCardProcessor()
result = processor.process_image_with_database(image, session_id)
```

#### Breaking Changes

**Configuration System**
- Environment variables now required
- Configuration class replaces hardcoded values
- Model paths must be specified in `.env`

**API Changes**
- New response format with confidence scores
- Added session tracking
- Enhanced error handling

**Database Integration**
- Optional MongoDB integration
- Structured result storage
- Metrics tracking

#### Compatibility Layer

Để maintain backward compatibility:

```python
# legacy_compatibility.py
from src.core.id_card_processor import IDCardProcessor
from src.models.model_manager import ModelManager

# Legacy function wrapper
def process_image(image_path: str):
    """Legacy compatibility function"""
    processor = IDCardProcessor()
    
    # Convert to new format
    from PIL import Image
    image = Image.open(image_path)
    
    result = processor.process_image_wtih_vietocr(image)
    
    # Convert to legacy format
    legacy_result = {
        'extracted_info': result[0] if isinstance(result, tuple) else result,
        'success': True
    }
    
    return legacy_result

# Export for backward compatibility
__all__ = ['process_image']
```

---

## 📚 Additional Resources

### 📖 Documentation
- [Project Structure Guide](docs/PROJECT_STRUCTURE.md)
- [Deployment Guide](docs/DEPLOYMENT_GUIDE.md)
- [Comprehensive Deployment Guide](docs/COMPREHENSIVE_DEPLOYMENT_GUIDE.md)
- [K3D Deployment Guide](docs/K3D-DEPLOYMENT.md)
- [Monitoring Documentation](docs/MONITORING_README.md)

### 🛠️ Development Tools
- **IDE Extensions**: Python, Docker, Kubernetes extensions
- **Development Dependencies**: pytest, black, flake8, mypy
- **Monitoring Tools**: Prometheus, Grafana, Loki stack

### 🏭 Production Considerations
- **Scaling**: Horizontal pod autoscaling với Kubernetes
- **Load Balancing**: Nginx hoặc cloud load balancers  
- **Caching**: Redis cho model caching
- **CDN**: CloudFlare cho static assets
- **Database**: MongoDB cluster cho high availability

### 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### 📝 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### 🙏 Acknowledgments
- **VietOCR team** for Vietnamese text recognition
- **Ultralytics** for YOLO models
- **PaddleOCR** for text detection
- **Vietnamese OCR community** for research and development

---

## 🆘 Support

### 📞 Getting Help
- **Issues**: GitHub Issues for bugs và feature requests
- **Discussions**: GitHub Discussions for questions
- **Documentation**: Check docs/ folder for detailed guides

### 🐛 Bug Reports
When reporting bugs, include:
- Operating system và Python version
- Error messages và stack traces
- Steps to reproduce
- Expected vs actual behavior
- Configuration files (sanitized)

### 💡 Feature Requests
For feature requests, describe:
- Use case và motivation
- Proposed solution
- Alternative solutions considered
- Impact on existing functionality

---

**© 2024 Vietnamese ID Card OCR Project. All rights reserved.**

*This complete guide provides everything needed to successfully deploy, monitor, and maintain the Vietnamese ID Card OCR system. Follow the steps carefully and refer to specific sections for detailed implementation guidance.*
