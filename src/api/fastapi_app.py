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
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Response
import time
import psutil
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
from typing import Dict, Any, Optional
import json
from datetime import datetime
import os
import threading

from ..models.model_manager import ModelManager
from ..core.id_card_processor import IDCardProcessor
from ..webhooks.alert_handlers import router as alert_router


# Enhanced Metrics for Comprehensive Monitoring
REQUEST_COUNT = Counter('request_count_total', 'Total requests processed')
PROCESSING_TIME = Histogram(
    'processing_time_seconds', 'Time spent processing request',
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
)
ERROR_COUNT = Counter('error_count_total', 'Total errors encountered')
SUCCESS_COUNT = Counter('success_count_total', 'Total successful requests')

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.FileHandler('logs/error.log', mode='a'),
        logging.StreamHandler()
    ]
)

# Create specialized loggers
logger = logging.getLogger(__name__)
error_logger = logging.getLogger('error')
model_logger = logging.getLogger('model')
metrics_logger = logging.getLogger('metrics')

# Warn if GPU monitoring is not available
if not GPU_AVAILABLE:
    logger.warning("GPUtil not available. GPU metrics will be disabled.")

# Set error logger to only log errors to error.log
error_handler = logging.FileHandler('logs/error.log')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)

# Set model logger for model-specific logs
model_handler = logging.FileHandler('logs/model.log')
model_logger.addHandler(model_handler)

# Set metrics logger for metrics-specific logs
metrics_handler = logging.FileHandler('logs/metrics.log')
metrics_logger.addHandler(metrics_handler)


class SystemMetricsCollector:
    """Collects system metrics for monitoring."""

    def __init__(self):
        self.is_running = False
        self.thread = None

    def start_collection(self):
        """Start collecting system metrics in background."""
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(
                target=self._collect_metrics, daemon=True)
            self.thread.start()
            logger.info("System metrics collection started")

    def stop_collection(self):
        """Stop collecting system metrics."""
        self.is_running = False
        if self.thread:
            self.thread.join()
        logger.info("System metrics collection stopped")

    def _collect_metrics(self):
        """Collect system metrics periodically."""
        while self.is_running:
            try:
                # CPU Usage
                cpu_percent = psutil.cpu_percent(interval=1)
                CPU_USAGE.set(cpu_percent)

                # Memory Usage
                memory = psutil.virtual_memory()
                MEMORY_USAGE.set(memory.percent)

                # Disk Usage
                disk = psutil.disk_usage('/')
                DISK_USAGE.labels(mountpoint='/').set(disk.percent)

                # Network Stats
                net_io = psutil.net_io_counters()
                NETWORK_BYTES_SENT.inc(net_io.bytes_sent)
                # GPU Usage (if available)
                NETWORK_BYTES_RECV.inc(net_io.bytes_recv)
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        for i, gpu in enumerate(gpus):
                            GPU_USAGE.labels(gpu_id=str(i)).set(gpu.load * 100)
                            GPU_MEMORY_USAGE.labels(gpu_id=str(i)).set(
                                gpu.memoryUtil * 100)
                            GPU_TEMPERATURE.labels(
                                gpu_id=str(i)).set(gpu.temperature)
                    except Exception as e:
                        logger.debug(f"GPU metrics not available: {e}")
                else:
                    logger.debug(
                        "GPU monitoring disabled - GPUtil not available")

                # Log metrics
                metrics_logger.info(
                    f"CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

            time.sleep(10)  # Collect every 10 seconds


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
        # Setup routes
        self._setup_routes()

        # Setup startup and shutdown events
        self._setup_events()

        # Include webhook handlers
        self.app.include_router(alert_router)

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

                # Process image with detailed metrics
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
                    model_logger.info(
                        f"Successful prediction for {file.filename} in {processing_time:.3f}s")
                else:
                    ERROR_COUNT.inc()
                    error_logger.error(
                        f"Failed prediction for {file.filename}: {result.get('error', 'Unknown error')}")

                return {
                    "status": "success",
                    "processing_time": processing_time,
                    "inference_time": inference_time,
                    "filename": file.filename,
                    "result": result
                }

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
            return Response(
                generate_latest(),
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
            logger.info("Shutting down Vietnamese ID Card API")

            # Stop system metrics collection
            self.metrics_collector.stop_collection()

            logger.info("API shutdown completed")

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

        # Log to different loggers based on content
        if result and result.get('status') == 'success':
            logger.info(
                f"Prediction: {json.dumps(log_entry, ensure_ascii=False)}")
        else:
            error_logger.error(
                f"Failed prediction: {json.dumps(log_entry, ensure_ascii=False)}")

        # Log model-specific metrics
        model_logger.info(
            # Store in feature store (for demo purposes)
            f"Model inference - Time: {inference_time:.3f}s, Confidence: {confidence:.3f}, File: {filename}")
        self.feature_store[datetime.now().isoformat()] = log_entry

    def run(self, host: str = "0.0.0.0", port: int = 8080, metrics_port: int = 8000):
        """Run the FastAPI application."""
        self.start_time = time.time()

        # Start system metrics collection
        self.metrics_collector.start_collection()

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
