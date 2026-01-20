from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for person detectors.

    All detector implementations must inherit from this class and implement
    the predict() method. The class provides common functionality like
    filtering by confidence threshold and drawing bounding boxes.

    Attributes:
        conf_threshold (float): Minimum confidence score for detections
        person_class_id (int): COCO class ID for person (0)
    """

    def __init__(self, conf_threshold: float = 0.25):
        """
        Initialize the base detector.

        Args:
            conf_threshold: Confidence threshold for filtering detections (0.0-1.0)
        """
        self.conf_threshold = conf_threshold
        self.person_class_id = 0  # COCO dataset: class 0 is "person"

    @abstractmethod
    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run inference on a single frame.

        This method must be implemented by all subclasses. It should return
        detections in a standardized format.

        Args:
            frame: Input image in BGR format (OpenCV standard)

        Returns:
            List of detections, where each detection is a dict with keys:
                - 'bbox': [x1, y1, x2, y2] in absolute pixel coordinates
                - 'conf': confidence score (0.0-1.0)
                - 'class_id': class ID (should be 0 for person)
        """

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        Args:
            frame: Input image in BGR format
            detections: List of detections from predict()
            color: BGR color for bounding boxes (default: green)
            thickness: Line thickness for bounding boxes (default: 2)

        Returns:
            Annotated frame with drawn bounding boxes and confidence scores
        """
        annotated = frame.copy()

        for det in detections:
            bbox = det["bbox"]
            conf = det["conf"]

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            label = f"Person {conf:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(
                annotated, (x1, y1 - label_h - baseline - 5), (x1 + label_w, y1), color, -1  # Filled rectangle
            )

            # Draw label text
            cv2.putText(
                annotated, label, (x1, y1 - baseline - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1  # Black text
            )

        return annotated

    def filter_person_class(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter detections to keep only persons above confidence threshold.

        Args:
            detections: Raw list of detections

        Returns:
            Filtered list containing only person detections with conf >= threshold
        """
        return [d for d in detections if d["class_id"] == self.person_class_id and d["conf"] >= self.conf_threshold]

    def __repr__(self) -> str:
        """String representation of the detector."""
        return f"{self.__class__.__name__}(conf_threshold={self.conf_threshold})"
