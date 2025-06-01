"""
FastAPI endpoint for Vietnamese ID Card OCR API.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
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
from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta
import os
import uuid
import socket
from pydantic import BaseModel

from ..models.model_manager import ModelManager
from ..core.id_card_processor import IDCardProcessor
from src.database import MongoDBClient, OCRResult, UserSession

# Metrics
REQUEST_COUNT = Counter('request_count_total', 'Total requests processed')
PROCESSING_TIME = Histogram('processing_time_seconds', 'Time spent processing request')
ERROR_COUNT = Counter('error_count_total', 'Total errors encountered')
SUCCESS_COUNT = Counter('success_count_total', 'Total successful requests')
BATCH_REQUEST_COUNT = Counter('batch_request_count_total', 'Total batch requests processed')

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

class ProcessingConfig(BaseModel):
    """Configuration for image processing."""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.3
    enhance_image: bool = True
    processing_method: str = "Auto (Gemini + OCR)"

class HistoryFilter(BaseModel):
    """Filter options for history endpoint."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    id_number: Optional[str] = None
    success_only: bool = False

def find_available_port(start_port: int, max_attempts: int = 100) -> int:
    """
    Find an available port starting from start_port.
    
    Args:
        start_port: The port to start checking from
        max_attempts: Maximum number of ports to check
        
    Returns:
        An available port number
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port after {max_attempts} attempts")

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
        async def process_id_card(
            file: UploadFile = File(...),
            config: ProcessingConfig = Depends()
        ) -> Dict[str, Any]:
            """
            Process Vietnamese ID card image and extract information.

            Args:
                file: Uploaded image file
                config: Processing configuration

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
                    
                    # Check for duplicate ID
                    id_number = result.get('extracted_info', {}).get('id_number')
                    if id_number:
                        duplicate_info = self._check_duplicate_id(id_number)
                        if duplicate_info.get('is_duplicate'):
                            result['duplicate_info'] = duplicate_info
                else:
                    ERROR_COUNT.inc()

                # Save to database
                session_id = str(uuid.uuid4())
                ocr_result = OCRResult(
                    session_id=session_id,
                    image_filename=file.filename,
                    extracted_info=result.get('extracted_info', {}),
                    processing_time=processing_time,
                    confidence_scores=result.get('confidence_scores', {}),
                    detected_text_regions=result.get('detected_regions', []),
                    success=True
                )
                
                result_id = self.db_client.save_ocr_result(ocr_result)
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

        @self.app.post("/process-batch/")
        async def process_batch(
            files: List[UploadFile] = File(...),
            config: ProcessingConfig = Depends(),
            max_batch_size: int = Query(5, ge=1, le=10)
        ) -> Dict[str, Any]:
            """
            Process multiple Vietnamese ID card images in batch.

            Args:
                files: List of uploaded image files
                config: Processing configuration
                max_batch_size: Maximum number of images to process

            Returns:
                JSON response with batch processing results
            """
            start_time = time.time()
            BATCH_REQUEST_COUNT.inc()

            try:
                # Limit batch size
                if len(files) > max_batch_size:
                    files = files[:max_batch_size]

                results = []
                for idx, file in enumerate(files):
                    try:
                        # Validate file type
                        if not file.content_type.startswith('image/'):
                            results.append({
                                'status': 'error',
                                'message': f"File {file.filename} must be an image"
                            })
                            continue

                        # Read and process image
                        contents = await file.read()
                        nparr = np.frombuffer(contents, np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if image is None:
                            results.append({
                                'status': 'error',
                                'message': f"Could not decode image file {file.filename}"
                            })
                            continue

                        # Process image
                        result = self.processor.process_id_card(image)
                        
                        # Check for duplicate ID
                        if result.get('status') == 'success':
                            id_number = result.get('extracted_info', {}).get('id_number')
                            if id_number:
                                duplicate_info = self._check_duplicate_id(id_number)
                                if duplicate_info.get('is_duplicate'):
                                    result['duplicate_info'] = duplicate_info

                        # Save to database
                        session_id = str(uuid.uuid4())
                        ocr_result = OCRResult(
                            session_id=session_id,
                            image_filename=file.filename,
                            extracted_info=result.get('extracted_info', {}),
                            processing_time=0.0,
                            confidence_scores=result.get('confidence_scores', {}),
                            detected_text_regions=result.get('detected_regions', []),
                            success=True
                        )
                        
                        result_id = self.db_client.save_ocr_result(ocr_result)
                        result['database_id'] = result_id
                        result['session_id'] = session_id

                        results.append(result)

                    except Exception as e:
                        logger.error(f"Error processing image {file.filename}: {str(e)}")
                        results.append({
                            'status': 'error',
                            'message': f"Error processing {file.filename}: {str(e)}"
                        })

                # Calculate total processing time
                processing_time = time.time() - start_time
                PROCESSING_TIME.observe(processing_time)

                return {
                    'status': 'success',
                    'total_images': len(files),
                    'processed_images': len(results),
                    'processing_time': processing_time,
                    'results': results
                }

            except Exception as e:
                ERROR_COUNT.inc()
                logger.error(f"Error processing batch: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Internal server error: {str(e)}"
                )

        @self.app.get("/history/{session_id}")
        async def get_processing_history(session_id: str):
            """Get processing history for a session."""
            try:
                results = self.db_client.get_ocr_results_by_session(session_id)
                return {
                    'status': 'success',
                    'session_id': session_id,
                    'total_results': len(results),
                    'results': results
                }
            except Exception as e:
                logger.error(f"Error retrieving history: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving history: {str(e)}"
                )

        @self.app.get("/search/{id_number}")
        async def search_by_id(id_number: str):
            """Search for ID card by ID number."""
            try:
                results = self.db_client.search_by_id_number(id_number)
                return {
                    'status': 'success',
                    'id_number': id_number,
                    'total_occurrences': len(results),
                    'results': results
                }
            except Exception as e:
                logger.error(f"Error searching by ID: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error searching by ID: {str(e)}"
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

        @self.app.get("/history")
        async def get_all_history(
            page: int = Query(1, ge=1, description="Page number"),
            page_size: int = Query(10, ge=1, le=100, description="Items per page"),
            filter: HistoryFilter = Depends()
        ):
            """
            Get all processing history with pagination and filtering.
            
            Args:
                page: Page number (starts from 1)
                page_size: Number of items per page (1-100)
                filter: Filter options
                    - start_date: Filter by start date
                    - end_date: Filter by end date
                    - id_number: Filter by ID number
                    - success_only: Only show successful results
            """
            try:
                # Build filter query
                query = {}
                
                # Date range filter
                if filter.start_date or filter.end_date:
                    date_filter = {}
                    if filter.start_date:
                        date_filter["$gte"] = filter.start_date
                    if filter.end_date:
                        date_filter["$lte"] = filter.end_date
                    if date_filter:
                        query["timestamp"] = date_filter
                
                # ID number filter
                if filter.id_number:
                    query["extracted_info.id_number"] = filter.id_number
                
                # Success filter
                if filter.success_only:
                    query["success"] = True
                
                # Get total count
                total_count = self.db_client._db[self.db_client.config.MONGODB_COLLECTION_RESULTS].count_documents(query)
                
                # Calculate pagination
                skip = (page - 1) * page_size
                total_pages = (total_count + page_size - 1) // page_size
                
                # Get paginated results
                results = list(self.db_client._db[self.db_client.config.MONGODB_COLLECTION_RESULTS]
                             .find(query)
                             .sort("timestamp", -1)
                             .skip(skip)
                             .limit(page_size))
                
                # Convert ObjectId to string and format datetime
                for result in results:
                    result["_id"] = str(result["_id"])
                    if "timestamp" in result:
                        result["timestamp"] = result["timestamp"].isoformat()
                
                return {
                    "status": "success",
                    "page": page,
                    "page_size": page_size,
                    "total_pages": total_pages,
                    "total_items": total_count,
                    "results": results
                }
                
            except Exception as e:
                logger.error(f"Error retrieving history: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving history: {str(e)}"
                )

    def _check_duplicate_id(self, id_number: str) -> Dict[str, Any]:
        """Check if ID number already exists in database."""
        try:
            results = self.db_client.search_by_id_number(id_number)
            if results:
                # Get the most recent result
                latest_result = results[0]
                return {
                    'is_duplicate': True,
                    'previous_result': latest_result,
                    'total_occurrences': len(results)
                }
            return {'is_duplicate': False}
        except Exception as e:
            logger.error(f"Error checking duplicate ID: {e}")
            return {'is_duplicate': False, 'error': str(e)}

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

        # Find available ports
        try:
            port = find_available_port(port)
            metrics_port = find_available_port(metrics_port)
            logger.info(f"Using port {port} for API server and port {metrics_port} for metrics server")
        except RuntimeError as e:
            logger.error(f"Failed to find available ports: {e}")
            raise

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
