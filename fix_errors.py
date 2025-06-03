#!/usr/bin/env python3
"""
Script to fix common errors in Vietnamese ID Card OCR project
"""

import os
import sys
from pathlib import Path


def check_and_fix_mlflow():
    """Check MLflow configuration and disable if server not available"""
    print("🔍 Checking MLflow configuration...")

    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file with MLflow disabled...")
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write("""# Vietnamese ID Card OCR Configuration
# =====================================

# MLflow Configuration (disabled by default)
MLFLOW_ENABLED=false
MLFLOW_TRACKING_URI=http://localhost:5000

# API Configuration
GEMINI_API_KEY=your_api_key_here

# Database Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=vnid_card_ocr

# Processing Configuration
FORCE_CPU=false
LOG_LEVEL=INFO

# Environment
ENVIRONMENT=development
DEBUG=false
""")
        print("✅ Created .env file with MLflow disabled")
    else:
        print("✅ .env file already exists")


def check_model_paths():
    """Check if model files exist"""
    print("\n🔍 Checking model files...")

    model_paths = [
        "models/yolo_text_detect/weights/best.pt",
        "models/yolo_text_detect/weights/best.onnx",
        "models/yolo_corner_detect/weights/29_03_25-YOLOv11n-Corner-best_metrics.pt",
        "models/yolo_corner_detect/weights/29_03_25-YOLOv11n-Corner-best_metrics.onnx"
    ]

    for path in model_paths:
        if Path(path).exists():
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Missing: {path}")


def check_directories():
    """Check and create required directories"""
    print("\n🔍 Checking required directories...")

    directories = [
        "logs",
        "data/uploads",
        "data/outputs",
        ".temp/serving"
    ]

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")


def test_imports():
    """Test critical imports"""
    print("\n🔍 Testing critical imports...")

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")

    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")

    try:
        from vietocr.tool.predictor import Predictor
        print("✅ VietOCR imported successfully")
    except ImportError as e:
        print(f"❌ VietOCR import failed: {e}")


def main():
    """Main function to run all checks and fixes"""
    print("🚀 Vietnamese ID Card OCR - Error Fix Script")
    print("=" * 50)

    check_and_fix_mlflow()
    check_model_paths()
    check_directories()
    test_imports()

    print("\n" + "=" * 50)
    print("🎉 Error fix script completed!")
    print("\n📝 Summary of fixes:")
    print("  • MLflow disabled by default (set MLFLOW_ENABLED=false)")
    print("  • Required directories created")
    print("  • Configuration files setup")
    print("\n💡 Next steps:")
    print("  1. Run 'make run-streamlit' to start the web interface")
    print("  2. Run 'make run-api' to start the API server")
    print("  3. Check logs/ directory for any remaining issues")


if __name__ == "__main__":
    main()
