from typing import Any, Dict, List, Tuple
import numpy as np

try:
    from ultralytics import YOLO, RTDETR
    ULTRALYTICS_AVIABLE = True
except ImportError:
    ULTRALYTICS_AVIABLE = False
    YOLO = None
    RTDETR = None

from .base_detector import BaseDetector

class UltralyticsDetector(BaseDetector):
    """
    Unified detector for Ultralytics models (YOLO and RT-DETR).

    This class wraps the Ultralytics API to provide a consistent interface.
    It supports multiple backends implicitly based on file extension:
        - .pt files: PyTorch backend
        - .engine files: TensorRT backend (optimized inference)

    Architectures supported:
        - YOLO
        - RT-DETR (Real-Time DEtection Transformer)
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold = 0.25,
        img_size: Tuple[int, int] = (640, 640),
        device: str = "cuda",
        model_type: str = "yolo"
    ):
        """
        Initialize the Ultralytics detector.

        Args:
            model_path: Path to the model weights (.pt or .engine file).
            conf_threshold: Confidence threshold for detections.
            img_size: Input image size (height, width).
            device: Target device ('cuda' or 'cpu').
            model_type: Type of model architecture ('yolo' or 'rtdetr').
        """
        super().__init__(conf_threshold)
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.img_size = img_size
        self.device = device

        if not ULTRALYTICS_AVIABLE:
            raise ImportError("Ultralytics library is not installed. Run: pip install ultralytics")
        
        if self.model_type == "rtdetr":
            print(f"🔄 Loading RT-DETR model: {model_path}")
            self.model = RTDETR(model_path)
        else:
            print(f"🔄 Loading YOLO model: {model_path}")
            self.model = YOLO(model_path)


    def predict(
        self,
        frame: np.ndarray    
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a single frame.

        Ultralytics automatically handles the backend inference
        (PyTorch or TensorRT) based on the loaded model file type.

        Args:
            frame: Input image in BGR format (OpenCV standard).

        Returns:
            List of detections filtered by person class.
        """
        results = self.model(
            frame,
            verbose=False,
            classes=[self.person_class_id],
            imgsz=self.img_size,
            device=self.device
        )

        detections = []

        for r in results:
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Get bb coord (xyxy format)
                    xyxy = box.xyxy[0].cpu().numpy()
                    # Get confidance score
                    conf = float(box.conf[0].cpu().numpy())
                    # Get class ID
                    cls = int(box.cls[0].cpu().numpy())

                    detections.append({
                        "bbox": xyxy.tolist(),
                        "conf": conf,
                        "class_id": cls
                    })

        return self.filter_person_class(detections)