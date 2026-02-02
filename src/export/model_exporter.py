import logging
from pathlib import Path

from ultralytics import RTDETR, YOLO


class ModelExporter:
    """
    Export utility for converting detection models to optimized formats.

    Supports:
    - ONNX: Cross-platform, CPU/GPU compatible
    - TensorRT: NVIDIA GPU optimized (FP16, INT8 with calibration)
    """

    @staticmethod
    def export_to_onnx(
        model_path: str,
        output_dir: str = "weights",
        opset: int = 17,
        simplify: bool = True,
        dynamic: bool = False,
        imgsz: int = 640,
    ) -> str:
        """
        Export PyTorch model to ONNX format.

        ONNX provides:
        - Cross-platform compatibility (Windows/Linux/macOS)
        - CPU and GPU support (CUDA)
        - 1.5-2x speedup vs PyTorch
        - Smaller file size

        Args:
            model_path: Path to PyTorch model (.pt file)
            output_dir: Directory to save exported model
            opset: ONNX opset version (default: 20)
            simplify: Simplify ONNX graph (default: True)
            dynamic: Dynamic batch size support (default: False)
            imgsz: Input image size (default: 640)

        Returns:
            Path to exported ONNX model

        Raises:
            FileNotFoundError: If source model doesn't exist
            RuntimeError: If export fails
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Source model not found: {model_path}")

        logging.info(f"Exporting to ONNX: {model_path}")
        logging.info(f"   Opset: {opset}")
        logging.info(f"   Input size: {imgsz}x{imgsz}")
        logging.info(f"   Simplify: {simplify}")

        try:
            # Load model based on type
            if "yolo" in model_path.lower():
                model = YOLO(model_path)
            else:
                model = RTDETR(model_path)

            # Export to ONNX
            onnx_path = model.export(format="onnx", opset=opset, simplify=simplify, dynamic=dynamic, imgsz=imgsz)

            # Get file size
            onnx_size = Path(onnx_path).stat().st_size / (1024 * 1024)

            logging.info(f"ONNX export successful!")
            logging.info(f"   Output: {onnx_path}")
            logging.info(f"   Size: {onnx_size:.2f} MB")

            return str(onnx_path)

        except Exception as e:
            logging.error(f"ONNX export failed: {e}")
            raise RuntimeError(f"ONNX export failed: {e}")

    @staticmethod
    def export_to_tensorrt(
        model_path: str,
        output_dir: str = "weights",
        fp16: bool = False,
        int8: bool = False,
        workspace: int = 4,
        imgsz: int = 640,
        data: str = None,
    ) -> str:
        """
        Export PyTorch model to TensorRT engine.

        Supports FP16 and INT8 (requires calibration data for INT8).

        Args:
            model_path: Path to PyTorch model (.pt file)
            output_dir: Directory to save exported engine
            fp16: Use FP16 precision (default: False)
            int8: Use INT8 quantization (default: False).
                  Note: INT8 requires a 'data' calibration file (YAML) for stable accuracy.
            workspace: GPU workspace size in GB (default: 4)
            imgsz: Input image size (default: 640)
            data: Path to dataset config (YAML) for INT8 calibration.

        Returns:
            Path to exported TensorRT engine

        Raises:
            FileNotFoundError: If source model doesn't exist
            RuntimeError: If export fails (no GPU, wrong CUDA version)
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Source model not found: {model_path}")

        # Determine precision for logs
        if int8:
            precision_str = "INT8"
            logging.warning("INT8 Mode Enabled.")
            logging.warning("This usually requires calibration data. If export fails, disable INT8.")
            if data is None:
                logging.warning("WARNING: No calibration data provided. Accuracy may drop or export may fail.")
        elif fp16:
            precision_str = "FP16"
        else:
            precision_str = "FP32"

        logging.info(f"Exporting to TensorRT: {model_path}")
        logging.info(f"   Precision: {precision_str}")
        logging.info(f"   Workspace: {workspace} GB")
        logging.info(f"   Input size: {imgsz}x{imgsz}")

        try:
            # Load model based on type
            if "yolo" in model_path.lower():
                model = YOLO(model_path)
            else:
                model = RTDETR(model_path)

            # Export arguments
            export_args = {
                "format": "engine",
                "half": fp16,
                "int8": int8,
                "opset": 17,
                "workspace": workspace,
                "dynamic": False,
                "imgsz": imgsz,
            }

            # Pass data if provided (required for INT8)
            if data:
                export_args["data"] = data
                logging.info(f"   Using calibration data: {data}")

            # Export to TensorRT
            trt_path = model.export(**export_args)

            # Get file size
            trt_size = Path(trt_path).stat().st_size / (1024 * 1024)

            logging.info(f"TensorRT export successful!")
            logging.info(f"   Output: {trt_path}")
            logging.info(f"   Size: {trt_size:.2f} MB")

            return str(trt_path)

        except Exception as e:
            # Provide helpful error messages for INT8
            if int8 and ("calibration" in str(e).lower() or "data" in str(e).lower()):
                logging.error(f"INT8 Export failed: {e}")
                raise RuntimeError(
                    f"INT8 Export failed: {e}\n"
                    "INT8 mode requires calibration data. "
                    "Please provide a YAML dataset config or disable int8."
                )

            logging.error(f"TensorRT export failed: {e}")
            raise RuntimeError(f"TensorRT export failed: {e}")

    @staticmethod
    def export_model(model_path: str, format: str, output_dir: str = "weights", **kwargs) -> str:
        """
        Universal export method supporting multiple formats.

        Args:
            model_path: Path to source PyTorch model
            format: Export format ('onnx' or 'tensorrt')
            output_dir: Output directory
            **kwargs: Format-specific arguments

        Returns:
            Path to exported model

        Raises:
            ValueError: If format is not supported
        """
        if format.lower() == "onnx":
            return ModelExporter.export_to_onnx(model_path, output_dir, **kwargs)
        elif format.lower() in ["tensorrt", "engine", "trt"]:
            return ModelExporter.export_to_tensorrt(model_path, output_dir, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'onnx' or 'tensorrt'")
