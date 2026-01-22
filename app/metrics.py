import time
from typing import Dict, Any


def collect_metrics() -> Dict[str, Any]:
    """Collect various metrics for the inference service"""
    return {
        "timestamp": time.time(),
        "requests_count": 0,
        "avg_latency": 0.0,
        "error_rate": 0.0
    }