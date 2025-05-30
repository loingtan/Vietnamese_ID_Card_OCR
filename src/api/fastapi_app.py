"""
FastAPI endpoint for Vietnamese ID Card OCR API.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import logging
from prometheus_client import Counter, Histogram, start_http_server, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from fastapi import Response
import time
from typing import Dict, Any, Optional
import json
from datetime import datetime
import os
import uuid

from ..models.model_manager import ModelManager
from ..core.id_card_processor import IDCardProcessor
from src.database import MongoDBClient, OCRResult


# Metrics
REQUEST_COUNT = Counter('request_count_total', 'Total requests processed')
PROCESSING_TIME = Histogram(
    'processing_time_seconds', 'Time spent processing request')
ERROR_COUNT = Counter('error_count_total', 'Total errors encountered')
SUCCESS_COUNT = Counter('success_count_total', 'Total successful requests')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MONGODB_URL = "mongodb://localhost:27017"  # URL kết nối MongoDB
MONGODB_DATABASE = "id_card_ocr"  # Tên database
MONGODB_COLLECTION_RESULTS = "ocr_results"  # Collection lưu kết quả


class IDCardAPI:
    """FastAPI application for ID Card OCR."""

    def __init__(self, api_key: Optional[str] = None):
        self.app = FastAPI(
            title="Vietnamese ID Card Scanner API",
            description="API for scanning and extracting information from Vietnamese ID cards",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )

        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Initialize models
        self.model_manager = ModelManager(api_key=api_key)
        self.processor = IDCardProcessor(self.model_manager)

        # Feature store (simple in-memory cache for demo)
        self.feature_store = {}

        # Khởi tạo MongoDB client
        self.db_client = MongoDBClient()
        self.db_client.connect()

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.post("/process-id-card/")
        async def process_id_card(file: UploadFile = File(...)) -> Dict[str, Any]:
            """
            Process Vietnamese ID card image and extract information.

            Args:
                file: Uploaded image file

            Returns:
                JSON response with extracted information
            """
            start_time = time.time()
            REQUEST_COUNT.inc()

            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    ERROR_COUNT.inc()
                    raise HTTPException(
                        status_code=400,
                        detail="File must be an image"
                    )

                # Read and process image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    ERROR_COUNT.inc()
                    raise HTTPException(
                        status_code=400,
                        detail="Could not decode image file"
                    )

                # Process image
                result = self.processor.process_id_card(image)

                # Log prediction
                self._log_prediction(result, file.filename)

                # Calculate processing time
                processing_time = time.time() - start_time
                PROCESSING_TIME.observe(processing_time)

                if result.get('status') == 'success':
                    SUCCESS_COUNT.inc()
                else:
                    ERROR_COUNT.inc()

                # Sau khi xử lý ảnh và có kết quả
                session_id = str(uuid.uuid4())
                
                # Tạo OCRResult object
                ocr_result = OCRResult(
                    session_id=session_id,
                    image_filename=file.filename,
                    extracted_info=result.get('extracted_info', {}),
                    processing_time=processing_time,
                    confidence_scores=result.get('confidence_scores', {}),
                    detected_text_regions=result.get('detected_regions', []),
                    success=True
                )
                
                # Lưu vào database
                result_id = self.db_client.save_ocr_result(ocr_result)
                
                # Thêm ID vào kết quả trả về
                result['database_id'] = result_id
                result['session_id'] = session_id
                
                return result

            except HTTPException:
                raise
            except Exception as e:
                ERROR_COUNT.inc()
                logger.error(f"Error processing image: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error: {str(e)}"
                )

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_loaded": len(self.model_manager.models),
                "version": "1.0.0"
            }

        @self.app.get("/metrics")
        async def get_metrics():
            """Prometheus metrics endpoint."""
            # Create a new registry
            custom_registry = CollectorRegistry()

            # Unregister the duplicate metrics
            custom_registry.unregister('request_count')
            custom_registry.unregister('request_count_total')
            custom_registry.unregister('request_count_created')

            # Now you can register your metrics
            return Response(
                generate_latest(custom_registry),
                media_type=CONTENT_TYPE_LATEST
            )

        @self.app.get("/stats")
        async def get_stats():
            """Get current API statistics."""
            return {
                "total_requests": REQUEST_COUNT._value.get(),
                "successful_requests": SUCCESS_COUNT._value.get(),
                "failed_requests": ERROR_COUNT._value.get(),
                "models_loaded": list(self.model_manager.models.keys()),
                "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0
            }

        @self.app.get("/models")
        async def get_model_info():
            """Get information about loaded models."""
            return {
                "loaded_models": list(self.model_manager.models.keys()),
                "device": self.model_manager.get_device(),
                "model_details": {
                    name: "loaded" if model is not None else "failed"
                    for name, model in self.model_manager.models.items()
                }
            }

        @self.app.post("/reload-models")
        async def reload_models(model_name: Optional[str] = None):
            """Reload specific model or all models."""
            try:
                if model_name:
                    self.model_manager.reload_model(model_name)
                    return {"status": "success", "message": f"Model {model_name} reloaded"}
                else:
                    # Reload all models
                    self.model_manager._load_all_models()
                    return {"status": "success", "message": "All models reloaded"}
            except Exception as e:
                logger.error(f"Error reloading models: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to reload models: {str(e)}"
                )

    def _log_prediction(self, result: Dict[str, Any], filename: str):
        """Log prediction results for monitoring."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "result": result,
            "model_version": "1.0.0"
        }

        # Log to file
        logger.info(f"Prediction: {json.dumps(log_entry, ensure_ascii=False)}")

        # Store in feature store (for demo purposes)
        self.feature_store[datetime.now().isoformat()] = log_entry

    def run(self, host: str = "0.0.0.0", port: int = 8080, metrics_port: int = 8000):
        """Run the FastAPI application."""
        self.start_time = time.time()

        # Start Prometheus metrics server
        try:
            start_http_server(metrics_port)
            logger.info(f"Metrics server started on port {metrics_port}")
        except Exception as e:
            logger.warning(f"Could not start metrics server: {e}")

        # Start FastAPI server
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)


def create_app(api_key: Optional[str] = None) -> FastAPI:
    """Factory function to create FastAPI app."""
    api = IDCardAPI(api_key=api_key)
    return api.app


def main():
    """Main entry point for the API."""
    # Get API key from environment variable
    api_key = os.getenv("GEMINI_API_KEY")

    # Create and run the API
    api = IDCardAPI(api_key=api_key)
    api.run()


if __name__ == "__main__":
    main()
