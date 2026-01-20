from typing import Any, Dict, List, Tuple

import numpy as np
from ultralytics import RTDETR

from .base_detector import BaseDetector


class RTDETRDetector(BaseDetector):
    """
    RT-DETR (Real-Time DEtection TRansformer) object detector.

    This detector wraps the Ultralytics RT-DETR models and provides a
    standardized interface. RT-DETR uses transformer architecture with
    self-attention mechanisms instead of traditional CNN approaches.

    Architecture differences from YOLO:
        - Uses encoder-decoder transformer architecture
        - No NMS required (set prediction with Hungarian matching)
        - Better at handling occlusions and dense scenes
        - Slightly slower but more accurate than YOLO
    """

    def __init__(
        self, model_path: str = "rtdetr-l.pt", conf_threshold: float = 0.25, img_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize RT-DETR detector.

        Args:
            model_path: Path to RT-DETR weights file (.pt format)
            conf_threshold: Confidence threshold for detections
            img_size: Input image size as (height, width) tuple
        """
        super().__init__(conf_threshold)
        self.model = RTDETR(model_path)
        self.model_name = model_path
        self.img_size = img_size
        print(f"✅ Loaded RT-DETR model: {model_path}")

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run RT-DETR inference on a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of person detections with bbox, confidence, and class_id
        """
        # Run inference with person class filter (class 0 in COCO)
        results = self.model(
            frame,
            verbose=False,
            classes=[self.person_class_id],
            imgsz=self.img_size,
        )

        detections = []

        # Parse RT-DETR results
        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    detections.append(
                        {
                            "bbox": xyxy.tolist(),
                            "conf": conf,
                            "class_id": cls,
                        }
                    )

        # Filter by confidence threshold
        return self.filter_person_class(detections)
