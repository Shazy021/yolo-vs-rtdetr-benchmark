from typing import Tuple
from .yolo_detector import YOLODetector
from .rtdetr_detector import RTDETRDetector
from .onnx_detector import ONNXDetector
from .trt_detector import TRTDetector


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

        # PyTorch Backend
        if backend == "pytorch":
            if model == "yolo":
                return YOLODetector(weights_path, conf_threshold, input_size)
            elif model == "rtdetr":
                return RTDETRDetector(weights_path, conf_threshold, input_size)

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

        # TensorRT Backend
        elif backend == "tensorrt":
            return TRTDetector(
                weights_path,
                model_type=model,
                conf_threshold=conf_threshold,
                img_size=input_size,
            )

        raise ValueError(f"Unsupported configuration: Model={model}, Backend={backend}")