from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import logging
from prometheus_client import Counter, Histogram, start_http_server, CollectorRegistry
import time
from typing import Dict, Any
import json
from datetime import datetime

# Create a custom registry
registry = CollectorRegistry()

# Khởi tạo FastAPI app
app = FastAPI(
    title="Vietnamese ID Card Scanner API",
    description="API for scanning and extracting information from Vietnamese ID cards",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics with custom registry
REQUEST_COUNT = Counter('request_count', 'Total requests processed', registry=registry)
PROCESSING_TIME = Histogram(
    'processing_time_seconds', 'Time spent processing request', registry=registry)
ERROR_COUNT = Counter('error_count', 'Total errors encountered', registry=registry)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Feature store (simple in-memory cache for demo)
feature_store = {}


@app.post("/process-id-card/")
async def process_id_card(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Process Vietnamese ID card image and extract information
    """
    start_time = time.time()
    REQUEST_COUNT.inc()

    try:
        # Read and process image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process image using existing function
        result = process_image(image)

        # Log prediction
        log_prediction(result)

        # Calculate processing time
        processing_time = time.time() - start_time
        PROCESSING_TIME.observe(processing_time)

        return {
            "status": "success",
            "processing_time": processing_time,
            "result": result
        }

    except Exception as e:
        ERROR_COUNT.inc()
        logger.error(f"Error processing image: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


def log_prediction(result: Dict[str, Any]):
    """
    Log prediction results for monitoring
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "result": result,
        "model_version": "1.0.0"  # Add your model version here
    }

    # Log to file
    logger.info(f"Prediction: {json.dumps(log_entry)}")

    # Store in feature store (for demo purposes)
    feature_store[datetime.now().isoformat()] = log_entry


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}


@app.get("/metrics")
async def get_metrics():
    """
    Get current metrics
    """
    return {
        "request_count": REQUEST_COUNT._value.get(),
        "error_count": ERROR_COUNT._value.get(),
        "processing_time_avg": PROCESSING_TIME.observe()
    }

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8000)
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8080)
