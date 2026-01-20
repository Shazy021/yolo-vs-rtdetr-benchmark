from typing import Any, Dict, List, Tuple

import numpy as np

from .base_detector import BaseDetector

try:
    from ultralytics import RTDETR, YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class TRTDetector(BaseDetector):
    """
    TensorRT detector for optimized GPU inference.

    This detector loads TensorRT engine files (.engine) exported from
    PyTorch models. Provides maximum performance on NVIDIA GPUs.

    Requirements:
        - NVIDIA GPU with CUDA support
        - TensorRT installed
        - Compatible engine file (built for same GPU architecture)
    """

    def __init__(
        self, engine_path: str, model_type: str = "yolo", conf_threshold: float = 0.25, img_size: Tuple = (640, 640)
    ):
        """
        Initialize TensorRT detector.

        Args:
            engine_path: Path to TensorRT engine file (.engine)
            model_type: Type of model - "yolo" or "rtdetr"
            conf_threshold: Confidence threshold for detections
            img_size: Input image size as (height, width) tuple

        Raises:
            ImportError: If ultralytics is not installed
            RuntimeError: If TensorRT engine cannot be loaded
        """
        super().__init__(conf_threshold)
        self.img_size = img_size

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics is not installed")

        # Load TensorRT engine through Ultralytics
        # Ultralytics automatically detects .engine format
        if model_type.lower() == "yolo":
            self.model = YOLO(engine_path, task="detect")
        elif model_type.lower() == "rtdetr":
            self.model = RTDETR(engine_path)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'yolo' or 'rtdetr'")

        self.model_name = engine_path
        self.model_type = model_type

        print(f"✅ Loaded TensorRT engine: {engine_path}")
        print(f"   Model type: {model_type.upper()}")

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run TensorRT inference on a single frame.

        TensorRT inference is typically 2-3x faster than PyTorch,
        especially with FP16 precision.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of person detections
        """
        # Run inference through Ultralytics API
        results = self.model(frame, verbose=False, classes=[self.person_class_id], imgsz=self.img_size)

        detections = []

        # Parse results (same format as PyTorch models)
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    detections.append({"bbox": xyxy.tolist(), "conf": conf, "class_id": cls})

        return self.filter_person_class(detections)
