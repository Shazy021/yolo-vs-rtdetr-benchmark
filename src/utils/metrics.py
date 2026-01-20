import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class MetricsTracker:
    """
    Track and analyze detection performance metrics with frame-level detail.

    Collects frame-level metrics during video processing and provides
    statistical analysis including percentiles and averages. Saves detailed
    data for later visualization and comparison.

    Attributes:
        name: Identifier for this metrics tracker
        latencies: List of inference latencies (ms) per frame
        detection_counts: List of detection counts per frame
        frame_count: Total frames processed
        timestamps: Processing timestamps for each frame
        start_time: Tracker initialization time
    """

    def __init__(self, name: str = "detector"):
        """
        Initialize metrics tracker.

        Args:
            name: Descriptive name for this tracker (e.g., "yolo_onnx")
        """
        self.name = name
        self.latencies = []  # milliseconds per frame
        self.detection_counts = []  # detections per frame
        self.timestamps = []  # relative timestamps
        self.frame_count = 0
        self.start_time = datetime.now()

    def add_frame(self, latency_ms: float, num_detections: int):
        """
        Add metrics for a single frame.

        Args:
            latency_ms: Inference latency in milliseconds
            num_detections: Number of objects detected
        """
        self.latencies.append(latency_ms)
        self.detection_counts.append(num_detections)
        self.timestamps.append((datetime.now() - self.start_time).total_seconds())
        self.frame_count += 1

    def get_summary(self) -> Dict[str, Any]:
        """
        Calculate summary statistics.

        Returns:
            Dictionary containing:
                - name: Tracker name
                - frame_count: Total frames processed
                - avg_latency_ms: Mean latency
                - median_latency_ms: Median latency
                - std_latency_ms: Standard deviation
                - p95_latency_ms: 95th percentile latency
                - p99_latency_ms: 99th percentile latency
                - min_latency_ms: Minimum latency
                - max_latency_ms: Maximum latency
                - avg_fps: Average FPS
                - avg_detections: Average detections per frame
                - total_detections: Total detections across all frames
                - processing_time: Total processing time (seconds)
        """
        if not self.latencies:
            return {}

        latencies = np.array(self.latencies)

        return {
            "name": self.name,
            "frame_count": self.frame_count,
            "avg_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "avg_fps": float(1000 / np.mean(latencies)) if np.mean(latencies) > 0 else 0,
            "avg_detections": float(np.mean(self.detection_counts)),
            "total_detections": int(np.sum(self.detection_counts)),
            "processing_time": self.timestamps[-1] if self.timestamps else 0,
        }

    def get_detailed_data(self) -> Dict[str, Any]:
        """
        Get detailed frame-by-frame data for visualization.

        Returns:
            Dictionary containing:
                - name: Tracker name
                - summary: Summary statistics
                - frames: List of per-frame data with:
                    - frame_num: Frame number
                    - latency_ms: Inference latency
                    - fps: Instantaneous FPS
                    - detections: Detection count
                    - timestamp: Relative timestamp (seconds)
        """
        summary = self.get_summary()

        frames_data = []
        for i, (latency, detections, timestamp) in enumerate(
            zip(self.latencies, self.detection_counts, self.timestamps)
        ):
            frames_data.append(
                {
                    "frame_num": i + 1,
                    "latency_ms": float(latency),
                    "fps": float(1000 / latency) if latency > 0 else 0,
                    "detections": int(detections),
                    "timestamp": float(timestamp),
                }
            )

        return {
            "name": self.name,
            "summary": summary,
            "frames": frames_data,
            "metadata": {"start_time": self.start_time.isoformat(), "total_frames": self.frame_count},
        }

    def save_to_file(self, filepath: str, detailed: bool = True):
        """
        Save metrics to JSON file.

        Args:
            filepath: Output JSON file path
            detailed: If True, saves frame-by-frame data; if False, only summary
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if detailed:
            data = self.get_detailed_data()
        else:
            data = self.get_summary()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"📊 Metrics saved to: {filepath}")

    def print_summary(self):
        """Print formatted metrics summary to console."""
        summary = self.get_summary()

        if not summary:
            print(f"⚠️  No metrics collected for {self.name}")
            return

        print(f"\n{'='*60}")
        print(f"📊 Metrics Summary: {self.name}")
        print(f"{'='*60}")
        print(f"Frames processed: {summary['frame_count']}")
        print(f"\nLatency Statistics (ms):")
        print(f"  Mean: {summary['avg_latency_ms']:.2f}")
        print(f"  Median: {summary['median_latency_ms']:.2f}")
        print(f"  Std Dev: {summary['std_latency_ms']:.2f}")
        print(f"  Min: {summary['min_latency_ms']:.2f}")
        print(f"  Max: {summary['max_latency_ms']:.2f}")
        print(f"  P95: {summary['p95_latency_ms']:.2f}")
        print(f"  P99: {summary['p99_latency_ms']:.2f}")
        print(f"\nPerformance:")
        print(f"  Average FPS: {summary['avg_fps']:.2f}")
        print(f"  Total time: {summary['processing_time']:.2f}s")
        print(f"\nDetections:")
        print(f"  Avg per frame: {summary['avg_detections']:.1f}")
        print(f"  Total: {summary['total_detections']}")
        print(f"{'='*60}")

    def get_latency_percentiles(self, percentiles: List[float] = None) -> Dict[float, float]:
        """
        Calculate custom latency percentiles.

        Args:
            percentiles: List of percentile values (0-100)

        Returns:
            Dictionary mapping percentile to latency value
        """
        if percentiles is None:
            percentiles = [50, 75, 90, 95, 99]

        if not self.latencies:
            return {}

        latencies = np.array(self.latencies)
        return {p: float(np.percentile(latencies, p)) for p in percentiles}
