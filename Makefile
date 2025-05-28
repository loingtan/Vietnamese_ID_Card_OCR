s# Vietnamese ID Card OCR Makefile

.PHONY: help install install-windows install-dev setup run-streamlit run-api test lint format clean docker-build docker-run

# Default target
help:
	@echo "Vietnamese ID Card OCR - Available commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install         Install package and dependencies"
	@echo "	 install-windows Install package and dependencies for Windows"
	@echo "  install-dev     Install package with development dependencies"
	@echo "  setup           Setup environment and validate models"
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
	docker build -t vnid-card-ocr .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Environment
create-env:
	@if not exist .env (copy .env.example .env)
	@echo "Environment file created. Please edit .env with your settings."
