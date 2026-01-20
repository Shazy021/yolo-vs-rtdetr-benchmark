from typing import Any, Dict, List, Tuple

import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector


class YOLODetector(BaseDetector):
    """
    YOLO object detector using Ultralytics PyTorch implementation.

    This detector wraps the Ultralytics YOLO models and
    provides a standardized interface through the BaseDetector class.
    """

    def __init__(
        self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.25, img_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO weights file (.pt format)
            conf_threshold: Confidence threshold for detections
            img_size: Input image size as (height, width) tuple
        """
        super().__init__(conf_threshold)
        self.model = YOLO(model_path)
        self.model_name = model_path
        self.img_size = img_size
        print(f"✅ Loaded YOLO model: {model_path}")

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLO inference on a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of person detections with bbox, confidence, and class_id
        """
        # Run inference with person class filter (class 0 in COCO)
        results = self.model(frame, verbose=False, classes=[self.person_class_id], imgsz=self.img_size)

        detections = []

        # Parse YOLO results
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                # Extract bounding boxes, confidences, and class IDs
                for box in boxes:
                    # Get bbox coordinates in xyxy format
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    detections.append({"bbox": xyxy.tolist(), "conf": conf, "class_id": cls})

        # Filter by confidence threshold
        return self.filter_person_class(detections)
