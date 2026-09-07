"""
Runtime operational metrics tracker for CreditRiskML.
Tracks request counts, success/failure rates, and latency percentiles (p50, p95, p99).
"""
import time
import numpy as np
from typing import Dict, Any, List

class MetricsTracker:
    def __init__(self):
        self.total_requests: int = 0
        self.successful_predictions: int = 0
        self.failed_predictions: int = 0
        self.latencies: List[float] = []

    def record_request(self, latency_ms: float, success: bool = True):
        self.total_requests += 1
        if success:
            self.successful_predictions += 1
        else:
            self.failed_predictions += 1

        self.latencies.append(latency_ms)
        # Keep window bounded to last 5000 requests
        if len(self.latencies) > 5000:
            self.latencies = self.latencies[-5000:]

    def get_metrics(self) -> Dict[str, Any]:
        if not self.latencies:
            return {
                "total_requests": self.total_requests,
                "successful_predictions": self.successful_predictions,
                "failed_predictions": self.failed_predictions,
                "latency_avg_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0
            }

        arr = np.array(self.latencies)
        return {
            "total_requests": self.total_requests,
            "successful_predictions": self.successful_predictions,
            "failed_predictions": self.failed_predictions,
            "latency_avg_ms": round(float(np.mean(arr)), 2),
            "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
            "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
            "latency_p99_ms": round(float(np.percentile(arr, 99)), 2)
        }

metrics_tracker = MetricsTracker()
