from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  onnxruntime not installed. Run: pip install onnxruntime-gpu")

from .base_detector import BaseDetector


class ONNXDetector(BaseDetector):
    """
    Universal ONNX Runtime detector for YOLO and RT-DETR models.

    Provides cross-platform inference with CPU and CUDA GPU support.
    Includes built-in NMS (Non-Maximum Suppression) for YOLO models.

    Attributes:
        nms_threshold: IoU threshold for NMS filtering
        model_type: Model architecture ('yolo' or 'rtdetr')
        input_h: Model input height in pixels
        input_w: Model input width in pixels
        device: Active execution device ('GPU (CUDA)' or 'CPU')
    """

    def __init__(
        self,
        onnx_path: str,
        use_gpu: bool = True,
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        model_type: str = "yolo",
        input_size: Tuple[int, int] = None,
    ):
        """
        Initialize ONNX detector.

        Args:
            onnx_path: Path to ONNX model file (.onnx)
            use_gpu: Use CUDA GPU if available (default: True)
            conf_threshold: Confidence threshold for detections (0.0-1.0)
            nms_threshold: IoU threshold for NMS (0.0-1.0, default: 0.45)
            model_type: Model architecture - 'yolo' or 'rtdetr'
            input_size: Custom input size as (height, width) tuple,
                       None to auto-detect from model

        Raises:
            ImportError: If onnxruntime is not installed
        """
        super().__init__(conf_threshold)

        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed")

        self.nms_threshold = nms_threshold
        self.model_type = model_type.lower()

        # Configure execution providers (GPU first if available)
        providers = []
        if use_gpu and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        # Create inference session
        self.session = ort.InferenceSession(onnx_path, providers=providers)

        # Get model I/O metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape_info = self.session.get_inputs()[0].shape

        # Parse or use custom input size
        if input_size is not None:
            self.input_h, self.input_w = input_size
            print(f"   Using custom input size: {self.input_w}x{self.input_h}")
        else:
            self.input_h, self.input_w = self._parse_input_shape(self.input_shape_info)

        self.output_names = [output.name for output in self.session.get_outputs()]
        self.output_shapes = [output.shape for output in self.session.get_outputs()]
        self.device = "GPU (CUDA)" if "CUDAExecutionProvider" in self.session.get_providers() else "CPU"

        # Print initialization summary
        print(f"✅ Loaded ONNX model: {onnx_path}")
        print(f"   Model type: {self.model_type.upper()}")
        print(f"   Device: {self.device}")
        print(f"   Input shape: {self.input_shape_info}")
        print(f"   Parsed size: {self.input_h}x{self.input_w}")
        print(f"   Output shapes: {self.output_shapes}")

    def _parse_input_shape(self, shape_info) -> Tuple[int, int]:
        """
        Parse input shape from ONNX model metadata.

        Handles dynamic dimensions (represented as strings) by falling
        back to default 640x640.

        Args:
            shape_info: Shape info from ONNX model input (e.g., [1, 3, 640, 640])

        Returns:
            Tuple of (height, width) for model input
        """
        try:
            if len(shape_info) == 4:
                h, w = shape_info[2], shape_info[3]

                # Handle dynamic dimensions (strings like 'height', 'width')
                if isinstance(h, str) or isinstance(w, str):
                    return 640, 640

                return int(h), int(w)
        except (TypeError, IndexError):
            pass

        return 640, 640

    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess frame with letterbox resize.

        Performs aspect-ratio-preserving resize with padding (letterbox)
        and normalizes pixel values to [0, 1] range.

        Args:
            frame: Input BGR image from OpenCV

        Returns:
            Tuple containing:
                - input_tensor: Preprocessed tensor [1, 3, H, W] as float32
                - scale: Scaling factor applied during resize
                - padding: (pad_w, pad_h) padding added to each side
        """
        img_h, img_w = frame.shape[:2]

        # Calculate scale to fit image in target size
        scale = min(self.input_w / img_w, self.input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        # Resize image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas (gray fill, value 114)
        padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        pad_w = (self.input_w - new_w) // 2
        pad_h = (self.input_h - new_h) // 2
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        # Normalize [0, 255] -> [0, 1] and convert to CHW format
        input_tensor = padded.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))  # HWC -> CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension

        return input_tensor, scale, (pad_w, pad_h)

    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray) -> List[int]:
        """
        Perform Non-Maximum Suppression on bounding boxes.

        Filters overlapping detections by keeping only the highest
        confidence box when IoU exceeds the NMS threshold.

        Args:
            boxes: Bounding boxes in xyxy format [N, 4]
            scores: Confidence scores [N]

        Returns:
            List of indices to keep after NMS
        """
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

            # Keep boxes with IoU <= threshold
            inds = np.where(iou <= self.nms_threshold)[0]
            order = order[inds + 1]

        return keep

    def postprocess_yolo(
        self, outputs: List[np.ndarray], original_shape: Tuple[int, int], scale: float, padding: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Postprocess YOLO model outputs.

        Handles YOLOv8 output format [1, 84, N] or [1, N, 84],
        applies confidence filtering and NMS.

        Args:
            outputs: Raw ONNX outputs, shape [1, 84, N] or [1, N, 84]
            original_shape: Original image (height, width)
            scale: Preprocessing scale factor
            padding: Preprocessing padding (pad_w, pad_h)

        Returns:
            List of detections with bbox, conf, and class_id
        """
        output = outputs[0]

        # Remove batch dimension
        if len(output.shape) == 3:
            output = output[0]

        # Handle [84, N] format -> transpose to [N, 84]
        if output.shape[0] == 84:
            output = output.T

        # Split output into boxes and class scores
        boxes = output[:, :4]
        class_scores = output[:, 4:]
        person_scores = class_scores[:, self.person_class_id]

        # Filter by confidence threshold
        mask = person_scores >= self.conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = person_scores[mask]

        if len(filtered_boxes) == 0:
            return []

        # Convert center format (xywh) to corner format (xyxy)
        cx, cy, w, h = filtered_boxes.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        # Apply NMS
        keep_indices = self.non_max_suppression(boxes_xyxy, filtered_scores)

        return self._boxes_to_detections(
            boxes_xyxy[keep_indices], filtered_scores[keep_indices], original_shape, scale, padding
        )

    def postprocess_rtdetr(
        self, outputs: List[np.ndarray], original_shape: Tuple[int, int], scale: float, padding: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Postprocess RT-DETR model outputs.

        RT-DETR outputs normalized coordinates in [1, 300, 84] format.
        No NMS required as transformer uses set prediction with Hungarian matching.

        Args:
            outputs: Raw ONNX outputs, shape [1, 300, 84]
            original_shape: Original image (height, width)
            scale: Preprocessing scale factor
            padding: Preprocessing padding (pad_w, pad_h)

        Returns:
            List of detections with bbox, conf, and class_id
        """
        output = outputs[0]

        # Remove batch dimension
        if len(output.shape) == 3:
            output = output[0]  # [300, 84]

        # Split output into boxes and class scores
        boxes = output[:, :4]  # [300, 4] - normalized cxcywh
        class_scores = output[:, 4:]  # [300, 80] - class probabilities

        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence and person class
        mask = (max_scores >= self.conf_threshold) & (class_ids == self.person_class_id)
        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]

        if len(filtered_boxes) == 0:
            return []

        # Convert normalized cxcywh to xyxy in pixel coordinates
        cx, cy, w, h = filtered_boxes.T

        # Normalized cxcywh -> normalized xyxy
        x1_norm = cx - w / 2
        y1_norm = cy - h / 2
        x2_norm = cx + w / 2
        y2_norm = cy + h / 2

        # Denormalize to input size
        x1 = x1_norm * self.input_w
        y1 = y1_norm * self.input_h
        x2 = x2_norm * self.input_w
        y2 = y2_norm * self.input_h

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

        return self._boxes_to_detections(boxes_xyxy, filtered_scores, original_shape, scale, padding)

    def _boxes_to_detections(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        original_shape: Tuple[int, int],
        scale: float,
        padding: Tuple[int, int],
    ) -> List[Dict[str, Any]]:
        """
        Convert boxes to final detection format.

        Maps boxes from model input space back to original image
        coordinates by removing padding and scaling.

        Args:
            boxes: Bounding boxes in xyxy format [N, 4]
            scores: Confidence scores [N]
            original_shape: Original image (height, width)
            scale: Preprocessing scale factor
            padding: Preprocessing padding (pad_w, pad_h)

        Returns:
            List of detection dicts with bbox, conf, and class_id
        """
        pad_w, pad_h = padding
        img_h, img_w = original_shape

        detections = []
        for box, conf in zip(boxes, scores):
            x1, y1, x2, y2 = box

            # Remove padding and scale back to original coordinates
            x1 = (x1 - pad_w) / scale
            y1 = (y1 - pad_h) / scale
            x2 = (x2 - pad_w) / scale
            y2 = (y2 - pad_h) / scale

            # Clip to image boundaries
            x1 = np.clip(x1, 0, img_w)
            y1 = np.clip(y1, 0, img_h)
            x2 = np.clip(x2, 0, img_w)
            y2 = np.clip(y2, 0, img_h)

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(conf),
                    "class_id": self.person_class_id,
                }
            )

        return detections

    def predict(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run ONNX inference on a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of person detections with bbox, conf, and class_id
        """
        original_shape = frame.shape[:2]

        # Preprocess frame
        input_tensor, scale, padding = self.preprocess(frame)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess based on model type
        if self.model_type == "yolo":
            detections = self.postprocess_yolo(outputs, original_shape, scale, padding)
        else:
            detections = self.postprocess_rtdetr(outputs, original_shape, scale, padding)

        return detections
