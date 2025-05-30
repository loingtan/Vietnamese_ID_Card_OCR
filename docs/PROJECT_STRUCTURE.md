# Vietnamese ID Card OCR - Project Structure

This document describes the organized file and folder structure of the Vietnamese ID Card OCR project.

## 📁 Root Directory Structure

```
VnId-Card/
├── 📄 README.md                    # Main project documentation
├── 📄 requirements.txt             # Python dependencies (Linux/macOS)
├── 📄 requirements_windows.txt     # Python dependencies (Windows)
├── 📄 setup.py                     # Package installation configuration
├── 📄 Makefile                     # Development commands and automation
├── 📄 .env                         # Environment variables (not in git)
├── 📄 .gitignore                   # Git ignore rules
├── 📄 app.py                       # Legacy entry point (backward compatibility)
├── 📄 api_app.py                   # FastAPI application entry point
├── 📄 streamlit_app.py             # Streamlit web UI entry point
│
├── 📁 src/                         # Main source code
├── 📁 config/                      # Configuration files
├── 📁 data/                        # Data files (models, dictionaries, samples)
├── 📁 deployment/                  # Deployment configurations
├── 📁 scripts/                     # Utility scripts
├── 📁 tests/                       # Test files
├── 📁 docs/                        # Documentation
├── 📁 monitoring/                  # Monitoring and logging configurations
└── 📁 archive/                     # Archived/backup files
```

## 📂 Source Code Structure (`src/`)

```
src/
├── 📄 __init__.py                  # Package initialization
├── 📄 config.py                    # Legacy configuration (backward compatibility)
│
├── 📁 api/                         # REST API implementation
│   ├── 📄 __init__.py
│   └── 📄 fastapi_app.py          # FastAPI application and routes
│
├── 📁 core/                        # Core business logic
│   ├── 📄 __init__.py
│   └── 📄 id_card_processor.py    # Main OCR processing pipeline
│
├── 📁 models/                      # AI model management
│   ├── 📄 __init__.py
│   └── 📄 model_manager.py        # Centralized model loading and management
│
├── 📁 utils/                       # Utility functions
│   ├── 📄 __init__.py
│   ├── 📄 image_processing.py     # Image processing utilities
│   └── 📄 text_processing.py      # Vietnamese text processing utilities
│
├── 📁 ui/                          # User interfaces
│   ├── 📄 __init__.py
│   └── 📄 streamlit_app.py        # Streamlit web interface
│
├── 📁 database/                    # Database operations
│   ├── 📄 __init__.py
│   ├── 📄 models.py               # Database models and schemas
│   └── 📄 mongodb.py              # MongoDB connection and operations
│
└── 📁 webhooks/                    # Webhook handlers
    └── 📄 alert_handlers.py       # Alert handling for monitoring
```

## 📂 Configuration Structure (`config/`)

```
config/
├── 📄 settings.py                  # Main configuration system
└── 📄 .env.example                # Environment variables template
```

## 📂 Data Structure (`data/`)

```
data/
├── 📁 models/                      # AI/ML model files
│   ├── 📁 corner_detection_model/ # YOLO corner detection model
│   │   └── 📁 weight/
│   │       └── 📄 29_03_25-YOLOv11n-Corner-best_metrics.pt
│   └── 📁 yolo_detect_text/       # YOLO text detection models
│       ├── 📄 best.pt
│       └── 📄 bestv2.pt
│
├── 📁 dictionary/                  # Vietnamese text dictionaries
│   └── 📁 dictionaries/
│       └── 📁 hongocduc/
│           └── 📄 words.txt
│
└── 📁 samples/                     # Sample images for testing
```

## 📂 Deployment Structure (`deployment/`)

```
deployment/
├── 📁 docker/                      # Docker configuration
│   ├── 📄 Dockerfile
│   ├── 📄 docker-compose.yml
│   └── 📄 docker-compose.override.yml
│
├── 📁 k8s/                         # Kubernetes manifests
│   ├── 📄 deployment.yaml
│   ├── 📄 service.yaml
│   ├── 📄 ingress.yaml
│   ├── 📄 autoscaling.yaml
│   ├── 📄 namespace-rbac.yaml
│   └── 📄 persistent-volumes.yaml
│
└── 📁 k3d/                         # K3D configuration
    ├── 📄 k3d-config.yaml
    └── 📄 Makefile.k3d
```

## 📂 Scripts Structure (`scripts/`)

```
scripts/
├── 📁 setup/                       # Setup and installation scripts
│   ├── 📄 check-prerequisites.ps1
│   └── 📄 quick-start.ps1
│
└── 📁 dev/                         # Development and deployment scripts
    ├── 📄 deploy-k3d.ps1
    ├── 📄 deploy-k3d.sh
    ├── 📄 monitor-k3d.ps1
    └── 📄 test-deployment.ps1
```

## 📂 Monitoring Structure (`monitoring/`)

```
monitoring/
├── 📄 docker-compose.monitoring.yml
├── 📄 start-monitoring.bat
├── 📄 stop-monitoring.bat
├── 📄 cleanup-logs.bat
│
├── 📁 prometheus/                  # Prometheus configuration
│   ├── 📄 prometheus.yml
│   └── 📄 alert-rules.yml
│
├── 📁 grafana/                     # Grafana dashboards and config
│   ├── 📁 dashboards/
│   └── 📁 provisioning/
│
├── 📁 loki/                        # Loki logging configuration
│   └── 📄 loki-config.yml
│
├── 📁 alertmanager/                # Alertmanager configuration
│   └── 📄 alertmanager.yml
│
└── 📁 fluent-bit/                  # Fluent Bit logging agent
    ├── 📄 fluent-bit.conf
    ├── 📄 parsers.conf
    └── 📁 scripts/
```

## 📂 Tests Structure (`tests/`)

```
tests/                              # Test files (to be implemented)
├── 📁 unit/                        # Unit tests
├── 📁 integration/                 # Integration tests
├── 📁 fixtures/                    # Test data and fixtures
└── 📄 conftest.py                  # Pytest configuration
```

## 📂 Documentation Structure (`docs/`)

```
docs/
├── 📄 DEPLOYMENT_GUIDE.md         # Deployment instructions
├── 📄 K3D-DEPLOYMENT.md          # K3D-specific deployment guide
└── 📄 MONITORING_README.md        # Monitoring setup and usage
```

## 🏗️ Key Design Principles

### 1. **Separation of Concerns**
- **`src/`**: All source code organized by functional area
- **`config/`**: Centralized configuration management
- **`deployment/`**: Infrastructure and deployment files
- **`data/`**: Models, dictionaries, and data files

### 2. **Environment-Based Configuration**
- Configuration through environment variables
- Template files for easy setup
- Support for development, staging, and production environments

### 3. **Modular Architecture**
- Clear module boundaries
- Dependency injection for models and services
- Easy to test and maintain components

### 4. **Production-Ready Structure**
- Proper logging and monitoring setup
- Health checks and metrics
- Scalable deployment configurations

### 5. **Developer Experience**
- Clear entry points for different interfaces
- Comprehensive documentation
- Automated setup scripts

## 🚀 Entry Points

| Purpose | Command | File |
|---------|---------|------|
| **Web UI** | `streamlit run streamlit_app.py` | `streamlit_app.py` |
| **REST API** | `python api_app.py` | `api_app.py` |
| **Legacy** | `python app.py` | `app.py` |
| **Development** | `make run-dev` | `Makefile` |
| **Docker** | `docker-compose up` | `deployment/docker/` |
| **K8s** | `kubectl apply -f deployment/k8s/` | `deployment/k8s/` |
| **K3D** | `make -f deployment/k3d/Makefile.k3d all` | `deployment/k3d/` |

## 📋 Migration Notes

This structure maintains backward compatibility with the original monolithic design while providing a clean, scalable foundation for future development. The original `app.py` functionality has been preserved and refactored into the new modular structure.

Key migration benefits:
- ✅ Easier maintenance and debugging
- ✅ Better separation of concerns
- ✅ Production-ready deployment options
- ✅ Comprehensive monitoring and logging
- ✅ Environment-based configuration
- ✅ Scalable architecture
