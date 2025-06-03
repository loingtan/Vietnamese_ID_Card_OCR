"""
System metrics collector for monitoring system resources.
"""

import psutil
import threading
import time
import logging
from typing import Dict, Any, Optional
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

logger = logging.getLogger(__name__)

class SystemMetricsCollector:
    """Collects system metrics for monitoring."""

    def __init__(self, interval: float = 5.0):
        """
        Initialize the metrics collector.
        
        Args:
            interval: Time interval between collections in seconds
        """
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._metrics: Dict[str, Any] = {}

    def start_collection(self):
        """Start collecting metrics in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Metrics collection already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._collect_metrics)
        self._thread.daemon = True
        self._thread.start()
        logger.info("Started system metrics collection")

    def stop_collection(self):
        """Stop collecting metrics."""
        if self._thread is None or not self._thread.is_alive():
            return

        self._stop_event.set()
        self._thread.join(timeout=self.interval * 2)
        logger.info("Stopped system metrics collection")

    def _collect_metrics(self):
        """Collect system metrics periodically."""
        while not self._stop_event.is_set():
            try:
                # CPU metrics
                self._metrics['cpu_percent'] = psutil.cpu_percent(interval=1)
                self._metrics['cpu_count'] = psutil.cpu_count()
                self._metrics['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None

                # Memory metrics
                memory = psutil.virtual_memory()
                self._metrics['memory'] = {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                }

                # Disk metrics
                disk = psutil.disk_usage('/')
                self._metrics['disk'] = {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }

                # GPU metrics if available
                if GPU_AVAILABLE:
                    try:
                        gpus = GPUtil.getGPUs()
                        self._metrics['gpu'] = [{
                            'id': gpu.id,
                            'name': gpu.name,
                            'load': gpu.load * 100,
                            'memory_total': gpu.memoryTotal,
                            'memory_used': gpu.memoryUsed,
                            'memory_free': gpu.memoryFree,
                            'temperature': gpu.temperature
                        } for gpu in gpus]
                    except Exception as e:
                        logger.error(f"Error collecting GPU metrics: {e}")
                        self._metrics['gpu'] = None

                # Network metrics
                net_io = psutil.net_io_counters()
                self._metrics['network'] = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }

                # Add timestamp
                self._metrics['timestamp'] = time.time()

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")

            time.sleep(self.interval)

    def get_metrics(self) -> Dict[str, Any]:
        """Get the latest collected metrics."""
        return self._metrics.copy()

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self._metrics.get('cpu_percent', 0.0)

    def get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        return self._metrics.get('memory', {}).get('percent', 0.0)

    def get_disk_usage(self) -> float:
        """Get current disk usage percentage."""
        return self._metrics.get('disk', {}).get('percent', 0.0)

    def get_gpu_usage(self) -> Optional[Dict[str, Any]]:
        """Get current GPU usage information."""
        return self._metrics.get('gpu') 