import time
from typing import Dict, Any
from collections import deque
import threading


class MetricsCollector:
    """Thread-safe metrics collector for the inference service"""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.lock = threading.Lock()
        
        # Initialize metrics
        self.requests_count = 0
        self.error_count = 0
        self.latencies = deque(maxlen=max_samples)
        self.start_time = time.time()
    
    def record_request(self, latency: float, is_error: bool = False):
        """Record a request with its latency and error status"""
        with self.lock:
            self.requests_count += 1
            if is_error:
                self.error_count += 1
            self.latencies.append(latency)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Collect various metrics for the inference service"""
        with self.lock:
            total_requests = self.requests_count
            total_errors = self.error_count
            
            # Calculate averages
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
            p95_latency = self._calculate_percentile(95) if self.latencies else 0.0
            p99_latency = self._calculate_percentile(99) if self.latencies else 0.0
            error_rate = total_errors / total_requests if total_requests > 0 else 0.0
            uptime = time.time() - self.start_time
            
        return {
            "timestamp": time.time(),
            "uptime_seconds": uptime,
            "requests_count": total_requests,
            "error_count": total_errors,
            "error_rate": error_rate,
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "p95_latency_ms": round(p95_latency * 1000, 2),
            "p99_latency_ms": round(p99_latency * 1000, 2),
            "active_requests": len(self.latencies),
            "samples_tracked": len(self.latencies)
        }
    
    def _calculate_percentile(self, percentile: float) -> float:
        """Calculate percentile of latencies"""
        if not self.latencies:
            return 0.0
        
        sorted_latencies = sorted(list(self.latencies))
        index = int(len(sorted_latencies) * (percentile / 100.0))
        index = min(index, len(sorted_latencies) - 1)  # Ensure we don't go out of bounds
        return sorted_latencies[index]


# Global metrics collector instance
metrics_collector = MetricsCollector()


def collect_metrics() -> Dict[str, Any]:
    """Collect various metrics for the inference service"""
    return metrics_collector.get_metrics()


def record_request_metric(latency: float, is_error: bool = False):
    """Record a request metric"""
    metrics_collector.record_request(latency, is_error)