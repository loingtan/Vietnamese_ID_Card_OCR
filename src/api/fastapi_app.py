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
import sys
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta
import os
import uuid
import socket
from pydantic import BaseModel
import threading
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try relative imports first, fallback to absolute imports for testing
try:
    from ..models.model_manager import ModelManager
    from ..core.id_card_processor import IDCardProcessor
    from ..webhooks.alert_handlers import router as alert_router
    from ..config import get_config
    from ..monitor import SystemMetricsCollector
    from ..database import MongoDBClient, OCRResult
except ImportError:
    from src.models.model_manager import ModelManager
    from src.core.id_card_processor import IDCardProcessor
    from src.webhooks.alert_handlers import router as alert_router
    from src.config import get_config
    from src.monitor import SystemMetricsCollector
    from src.database import MongoDBClient, OCRResult

# Load configuration
config = get_config()

# Fix Windows console encoding for Vietnamese characters
if sys.platform == "win32":
    try:
        # Set console to UTF-8 mode on Windows
        import locale
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except (locale.Error, ImportError):
        try:
            # Alternative approach for Windows
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        except Exception:
            # If all else fails, we'll rely on the safe logging approach
            pass

# Enhanced Metrics for Comprehensive Monitoring
REQUEST_COUNT = Counter('request_count_total', 'Total requests processed')
PROCESSING_TIME = Histogram(
    'processing_time_seconds', 'Time spent processing request',
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)
ERROR_COUNT = Counter('error_count_total', 'Total errors encountered')
SUCCESS_COUNT = Counter('success_count_total', 'Total successful requests')
BATCH_REQUEST_COUNT = Counter('batch_request_count_total', 'Total batch requests processed')

# Model Performance Metrics
INFERENCE_TIME = Histogram(
    'inference_time_seconds', 'Model inference time',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)
CONFIDENCE_SCORE = Histogram(
    'confidence_score', 'Model confidence score',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)
MODEL_LOADING_STATUS = Gauge(
    'model_loading_status', 'Model loading status (1=loaded, 0=failed)', ['model_name'])

# System Resource Metrics
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')
MEMORY_USAGE = Gauge('memory_usage_percent', 'Memory usage percentage')
DISK_USAGE = Gauge('disk_usage_percent',
                   'Disk usage percentage', ['mountpoint'])
GPU_USAGE = Gauge('gpu_usage_percent', 'GPU usage percentage', ['gpu_id'])
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_percent',
                         'GPU memory usage percentage', ['gpu_id'])
GPU_TEMPERATURE = Gauge('gpu_temperature_celsius',
                        'GPU temperature in Celsius', ['gpu_id'])

# Network Metrics
NETWORK_BYTES_SENT = Counter(
    'network_bytes_sent_total', 'Total network bytes sent')
NETWORK_BYTES_RECV = Counter(
    'network_bytes_recv_total', 'Total network bytes received')

# Enhanced Logging configuration with multiple handlers
os.makedirs('logs', exist_ok=True)

# Configure stream handler with UTF-8 encoding for Windows
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Configure file handlers with UTF-8 encoding
file_handler = logging.FileHandler('logs/api.log', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

error_file_handler = logging.FileHandler(
    'logs/error.log', mode='a', encoding='utf-8')
error_file_handler.setLevel(logging.ERROR)
error_file_handler.setFormatter(file_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, error_file_handler, console_handler]
)

# Create specialized loggers
logger = logging.getLogger(__name__)
error_logger = logging.getLogger('error')
model_logger = logging.getLogger('model')
metrics_logger = logging.getLogger('metrics')

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
        # Initialize system metrics collector
        self.metrics_collector = SystemMetricsCollector()

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
        )        # Initialize models
        self.model_manager = ModelManager(api_key=api_key)
        self.processor = IDCardProcessor(self.model_manager)

        # Feature store (simple in-memory cache for demo)
        self.feature_store = {}

        # Khởi tạo MongoDB client
        self.db_client = MongoDBClient()
        self.db_client.connect()

        # Setup routes
        self._setup_routes()

        # Setup startup and shutdown events
        self._setup_events()

        # Include webhook handlers
        self.app.include_router(alert_router)

    def _process_single_image(self, image: np.ndarray, filename: str, config: ProcessingConfig) -> Dict[str, Any]:
        """Process a single image with error handling."""
        try:
            start_time = time.time()
            result = self.processor.process_id_card(image)
            inference_time = time.time() - start_time
            
            if result and 'extracted_info' in result:
                # Log successful prediction
                self._log_prediction(result, filename, inference_time)
                
                # Check for duplicate ID
                id_number = result['extracted_info'].get('id_number')
                if id_number:
                    duplicate_info = self._check_duplicate_id(id_number)
                    result['duplicate_info'] = duplicate_info
                
                return {
                    'status': 'success',
                    'extracted_info': result['extracted_info'],
                    'message': 'Successfully processed',
                    'filename': filename,
                    'processing_time': inference_time
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Failed to extract information',
                    'filename': filename
                }
        except Exception as e:
            logger.error(f"Error processing image {filename}: {e}")
            return {
                'status': 'error',
                'message': f"Error: {str(e)}",
                'filename': filename
            }

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
                    )                # Read and process image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    ERROR_COUNT.inc()
                    raise HTTPException(
                        status_code=400,
                        detail="Could not decode image file"
                    )

                logger.info(f"Processing file: {file.filename}")
                inference_start = time.time()
                result = self.processor.process_id_card(image)
                inference_time = time.time() - inference_start

                # Record inference time
                INFERENCE_TIME.observe(inference_time)

                # Record confidence scores if available
                if result and 'confidence' in result:
                    confidence = result['confidence']
                    CONFIDENCE_SCORE.observe(confidence)

                    # Log low confidence predictions
                    if confidence < 0.6:
                        error_logger.warning(
                            f"Low confidence prediction: {confidence:.3f} for file {file.filename}"
                        )

                # Log prediction with detailed metrics
                self._log_prediction(result, file.filename, inference_time)

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
                    error_logger.error(
                        f"Failed prediction for {file.filename}: {result.get('error', 'Unknown error')}")

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
            config: ProcessingConfig = Depends()
        ) -> Dict[str, Any]:
            """
            Process multiple Vietnamese ID card images in parallel.

            Args:
                files: List of uploaded image files
                config: Processing configuration

            Returns:
                JSON response with extracted information for all images
            """
            start_time = time.time()
            BATCH_REQUEST_COUNT.inc()
            
            try:
                # Validate all files
                for file in files:
                    if not file.content_type.startswith('image/'):
                        ERROR_COUNT.inc()
                        raise HTTPException(
                            status_code=400,
                            detail=f"File {file.filename} must be an image"
                        )
                
                # Read all images
                images = []
                for file in files:
                    contents = await file.read()
                    nparr = np.frombuffer(contents, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if image is None:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Could not decode image {file.filename}"
                        )
                    images.append((image, file.filename))
                
                # Process images in parallel
                results = []
                total_images = len(images)
                
                # Automatically determine number of workers based on CPU cores
                max_workers = min(os.cpu_count() or 4, total_images)
                
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    future_to_image = {
                        executor.submit(self._process_single_image, image, filename, config): (image, filename)
                        for image, filename in images
                    }
                    
                    # Process completed tasks as they finish
                    for future in as_completed(future_to_image):
                        result = future.result()
                        results.append(result)
                        
                        # Log progress
                        logger.info(f"Completed processing {len(results)}/{total_images} images")
                
                # Calculate total processing time
                total_time = time.time() - start_time
                
                # Prepare response
                response = {
                    'status': 'success',
                    'total_images': total_images,
                    'processed_images': len(results),
                    'total_processing_time': total_time,
                    'results': results
                }
                
                SUCCESS_COUNT.inc()
                return response
                
            except Exception as e:
                ERROR_COUNT.inc()
                logger.error(f"Error in batch processing: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing batch: {str(e)}"
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
                else:                    # Reload all models
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

    def _log_prediction(self, result: Dict[str, Any], filename: str, inference_time: float):
        """Log prediction results for monitoring."""
        confidence = result.get('confidence', 0.0) if result else 0.0

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "inference_time": inference_time,
            "confidence": confidence,
            "result": result,
            "model_version": "1.0.0",
            "success": result.get('status') == 'success' if result else False
        }

        try:
            # Log to different loggers based on content
            if result and result.get('status') == 'success':
                logger.info(
                    f"Prediction: {json.dumps(log_entry, ensure_ascii=False, indent=None)}")
            else:
                error_logger.error(
                    f"Failed prediction: {json.dumps(log_entry, ensure_ascii=False, indent=None)}")

            # Log model-specific metrics
            model_logger.info(
                f"Model inference - Time: {inference_time:.3f}s, Confidence: {confidence:.3f}, File: {filename}")

        except UnicodeEncodeError as e:
            # Fallback logging with ASCII encoding if Unicode fails
            logger.warning(f"Unicode encoding error in logging: {e}")
            safe_log_entry = {
                "timestamp": log_entry["timestamp"],
                "filename": filename,
                "inference_time": inference_time,
                "confidence": confidence,
                "model_version": "1.0.0",
                "success": log_entry["success"],
                "unicode_error": "Vietnamese text removed due to encoding issues"
            }
            logger.info(
                f"Prediction (safe): {json.dumps(safe_log_entry, ensure_ascii=True)}")

        # Store in feature store (for demo purposes)
        self.feature_store[datetime.now().isoformat()] = log_entry

    def _setup_events(self):
        """Setup FastAPI startup and shutdown events."""

        @self.app.on_event("startup")
        async def startup_event():
            """Handle application startup."""
            logger.info("Starting Vietnamese ID Card API")

            # Start system metrics collection
            self.metrics_collector.start_collection()

            # Initialize model loading status metrics
            for model_name in self.model_manager.models:
                model = self.model_manager.models[model_name]
                status = 1 if model is not None else 0
                MODEL_LOADING_STATUS.labels(model_name=model_name).set(status)

            logger.info("API startup completed")

        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Handle application shutdown."""
            logger.info(
                # Stop system metrics collection
                "Shutting down Vietnamese ID Card API")
            self.metrics_collector.stop_collection()

            logger.info("API shutdown completed")

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

        try:
            uvicorn.run(self.app, host=host, port=port)
        finally:
            # Stop metrics collection on shutdown
            self.metrics_collector.stop_collection()


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
