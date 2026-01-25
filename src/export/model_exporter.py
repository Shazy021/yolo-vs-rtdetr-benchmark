from pathlib import Path

from ultralytics import RTDETR, YOLO


class ModelExporter:
    """
    Export utility for converting detection models to optimized formats.

    Supports:
    - ONNX: Cross-platform, CPU/GPU compatible
    - TensorRT: NVIDIA GPU optimized, best performance
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

        Example:
            >>> exporter = ModelExporter()
            >>> onnx_path = exporter.export_to_onnx('yolov8n.pt')
            >>> print(f"Exported to: {onnx_path}")
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Source model not found: {model_path}")

        print(f"\n🔄 Exporting to ONNX: {model_path}")
        print(f"   Opset: {opset}")
        print(f"   Input size: {imgsz}x{imgsz}")
        print(f"   Simplify: {simplify}")

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

            print(f"✅ ONNX export successful!")
            print(f"   Output: {onnx_path}")
            print(f"   Size: {onnx_size:.2f} MB")

            return str(onnx_path)

        except Exception as e:
            raise RuntimeError(f"ONNX export failed: {e}")

    @staticmethod
    def export_to_tensorrt(
        model_path: str, output_dir: str = "weights", fp16: bool = False, workspace: int = 4, imgsz: int = 640
    ) -> str:
        """
        Export PyTorch model to TensorRT engine.

        TensorRT provides:
        - Maximum performance on NVIDIA GPUs
        - 2-3x speedup vs PyTorch, 1.5x vs ONNX
        - FP16 precision support
        - Optimized memory usage

        Requirements:
        - NVIDIA GPU with CUDA
        - TensorRT installed
        - Matching CUDA version

        Args:
            model_path: Path to PyTorch model (.pt file)
            output_dir: Directory to save exported engine
            fp16: Use FP16 precision (default: False, ~2x faster when True)
            workspace: GPU workspace size in GB (default: 4)
            imgsz: Input image size (default: 640)

        Returns:
            Path to exported TensorRT engine

        Raises:
            FileNotFoundError: If source model doesn't exist
            RuntimeError: If export fails (no GPU, wrong CUDA version)

        Example:
            >>> exporter = ModelExporter()
            >>> trt_path = exporter.export_to_tensorrt('yolov8n.pt', fp16=True)
            >>> print(f"Exported to: {trt_path}")
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Source model not found: {model_path}")

        print(f"\n🔄 Exporting to TensorRT: {model_path}")
        print(f"   Precision: {'FP16' if fp16 else 'FP32'}")
        print(f"   Workspace: {workspace} GB")
        print(f"   Input size: {imgsz}x{imgsz}")
        print(f"   ⚠️  This may take several minutes...")

        try:
            # Load model based on type
            if "yolo" in model_path.lower():
                model = YOLO(model_path)
            else:
                model = RTDETR(model_path)

            # Export to TensorRT
            trt_path = model.export(
                format="engine", half=fp16, opset=17, workspace=workspace, dynamic=False, imgsz=imgsz
            )

            # Get file size
            trt_size = Path(trt_path).stat().st_size / (1024 * 1024)

            print(f"✅ TensorRT export successful!")
            print(f"   Output: {trt_path}")
            print(f"   Size: {trt_size:.2f} MB")

            return str(trt_path)

        except Exception as e:
            raise RuntimeError(
                f"TensorRT export failed: {e}\n"
                "Make sure you have:\n"
                "  - NVIDIA GPU\n"
                "  - CUDA installed\n"
                "  - TensorRT installed\n"
                "  - Matching CUDA/TensorRT versions"
            )

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
