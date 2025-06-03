#!/usr/bin/env python3
import sys
sys.path.append('.')

print("Testing individual imports from model_manager.py...")

# Test each import individually
imports_to_test = [
    "import torch",
    "from ultralytics import YOLO",
    "from vietocr.tool.predictor import Predictor",
    "from vietocr.tool.config import Cfg",
    "from transformers import pipeline",
    "from google import genai",
    "from pathlib import Path",
    "import os",
    "import logging"
]

for imp in imports_to_test:
    try:
        exec(imp)
        print(f"✓ {imp}")
    except Exception as e:
        print(f"✗ {imp} - Error: {e}")

print("\nTesting config import...")
try:
    from config.settings import get_config
    print("✓ Config import successful")
except Exception as e:
    print(f"✗ Config import failed: {e}")

print("\nTesting if the class definition is reached...")
try:
    # Read and execute the file step by step
    with open('src/models/model_manager.py', 'r') as f:
        content = f.read()

    # Create a namespace to execute in
    namespace = {}
    exec(content, namespace)

    if 'ModelManager' in namespace:
        print("✓ ModelManager class found in execution namespace")
    else:
        print("✗ ModelManager class not found in execution namespace")
        print("Available items:", [
              k for k in namespace.keys() if not k.startswith('_')])

except Exception as e:
    print(f"✗ Execution error: {e}")
    import traceback
    traceback.print_exc()
