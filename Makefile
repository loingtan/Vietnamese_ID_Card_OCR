# Vietnamese ID Card OCR Makefile

.PHONY: help install install-windows install-dev setup run-streamlit run-api test lint format clean docker-build docker-run deploy-k3d deploy-k8s start-monitoring stop-monitoring

# Default target
help:
	@echo "Vietnamese ID Card OCR - Available commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install         Install package and dependencies"
	@echo "  install-windows Install package and dependencies for Windows"
	@echo "  install-dev     Install package with development dependencies"
	@echo "  setup           Setup environment and validate models"
	@echo "  setup-config    Setup configuration files"
	@echo ""
	@echo "Running Applications:"
	@echo "  run-streamlit   Run Streamlit web interface"
	@echo "  run-api         Run FastAPI server"
	@echo ""
	@echo "Development:"
	@echo "  test            Run tests"
	@echo "  lint            Run linting checks"
	@echo "  format          Format code with black and isort"
	@echo "  clean           Clean build artifacts"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build    Build Docker image"
	@echo "  docker-run      Run with Docker Compose"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-k3d      Deploy to K3D cluster"
	@echo "  deploy-k8s      Deploy to Kubernetes"
	@echo ""
	@echo "Monitoring:"
	@echo "  start-monitoring Start monitoring stack"
	@echo "  stop-monitoring  Stop monitoring stack"
	@echo ""

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-windows:
	pip install -r requirements_windows.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,api,ui]"

setup:
	@echo "Setting up Vietnamese ID Card OCR..."
	@python -c "from src.config import Config; print('Validating setup...'); results = Config.validate_setup(); print('Model validation:', results)"
	@echo "Setup complete!"

# Configuration setup
setup-config:
	@echo "Setting up configuration files..."
	@if not exist .env (copy config\.env.example .env && echo "Created .env file from template. Please edit it with your settings.")
	@if not exist logs mkdir logs
	@if not exist data\uploads mkdir data\uploads
	@if not exist data\outputs mkdir data\outputs
	@echo "Configuration setup complete!"

# Running applications
run-streamlit:
	streamlit run streamlit_app.py

run-api:
	python api_app.py

# Development
test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Docker
docker-build:
	docker build -f deployment\docker\Dockerfile -t vnid-card-ocr .

docker-run:
	docker-compose -f deployment\docker\docker-compose.yml up -d

docker-stop:
	docker-compose -f deployment\docker\docker-compose.yml down

docker-logs:
	docker-compose logs -f

# Deployment
deploy-k3d:
	@echo "Deploying to K3D cluster..."
	make -f deployment\k3d\Makefile.k3d all

deploy-k8s:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f deployment\k8s\

undeploy-k8s:
	@echo "Removing from Kubernetes..."
	kubectl delete -f deployment\k8s\ --ignore-not-found=true

# Monitoring
start-monitoring:
	@echo "Starting monitoring stack..."
	docker-compose -f monitoring\docker-compose.monitoring.yml up -d

stop-monitoring:
	@echo "Stopping monitoring stack..."
	docker-compose -f monitoring\docker-compose.monitoring.yml down

# Project structure
show-structure:
	@echo "Current project structure:"
	@tree /F /A

# Validation
validate-structure:
	@echo "Validating project structure..."
	@if exist src\__init__.py (echo ✓ Source package structure OK) else (echo ✗ Missing src\__init__.py)
	@if exist config\settings.py (echo ✓ Configuration system OK) else (echo ✗ Missing config\settings.py)
	@if exist deployment\docker\Dockerfile (echo ✓ Docker configuration OK) else (echo ✗ Missing Docker configuration)
	@if exist deployment\k8s\deployment.yaml (echo ✓ K8s configuration OK) else (echo ✗ Missing K8s configuration)
	@echo "Structure validation complete!"

# Complete setup for new developers
first-time-setup: install-windows setup-config validate-structure
	@echo ""
	@echo "🎉 First-time setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env file with your API keys and settings"
	@echo "2. Run 'make run-streamlit' for web interface"
	@echo "3. Run 'make run-api' for REST API"
	@echo "4. Check docs\PROJECT_STRUCTURE.md for more information"
