from typing import Tuple

from .onnx_detector import ONNXDetector
from .ultralytics_detector import UltralyticsDetector


class DetectorFactory:
    """Factory class to create detector instances dynamically."""

    @staticmethod
    def create(
        model: str,
        backend: str,
        weights_path: str,
        conf_threshold: float,
        nms_threshold: float,
        input_size: Tuple[int, int],
        use_gpu: bool = True,
    ):
        """
        Create a detector instance based on model and backend type.

        Args:
            model: Model architecture ('yolo' or 'rtdetr')
            backend: Inference backend ('pytorch', 'onnx', 'tensorrt')
            weights_path: Path to model weights
            conf_threshold: Confidence threshold
            nms_threshold: NMS threshold (for ONNX/TensorRT)
            input_size: Tuple (height, width)
            use_gpu: Enable GPU acceleration

        Returns:
            Initialized detector instance

        Raises:
            ValueError: If model/backend combination is not supported
        """
        model = model.lower()
        backend = backend.lower()
        actual_device = "cuda" if use_gpu else "cpu"

        # PyTorch Backend
        if backend in ("pytorch", "tensorrt"):
            return UltralyticsDetector(
                model_path=weights_path,
                conf_threshold=conf_threshold,
                img_size=input_size,
                device=actual_device,
                model_type=model
            )

        # ONNX Runtime Backend
        elif backend == "onnx":
            return ONNXDetector(
                weights_path,
                use_gpu=use_gpu,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                model_type=model,
                input_size=input_size,
            )

        raise ValueError(f"Unsupported configuration: Model={model}, Backend={backend}")
