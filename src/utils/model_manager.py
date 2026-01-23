from pathlib import Path

try:
    from ultralytics import RTDETR, YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from ..export import ModelExporter


class ModelManager:
    """Handles model file existence checks and preparation."""

    def __init__(self, config):
        """
        Initialize Model Manager.

        Args:
            auto_download: If True, auto-download missing PyTorch models.
            auto_export: If True, auto-export to ONNX/TensorRT if needed.
        """
        self.auto_download = config.get("automation.auto_download", False)
        self.auto_export = config.get("automation.auto_export", False)

    def ensure_model(self, weights_path: str, model_type: str, backend: str) -> bool:
        """
        Ensure model file exists. Download/Export if necessary and allowed.

        Args:
            weights_path: Expected path to weights file
            model_type: 'yolo' or 'rtdetr'
            backend: 'pytorch', 'onnx', 'tensorrt'

        Returns:
            True if model is ready

        Raises:
            FileNotFoundError: If model not found and auto-download disabled
            RuntimeError: If download/export fails
        """
        path = Path(weights_path)

        if path.exists():
            return True

        print(f"⚠️  Model not found: {weights_path}")

        # Case 1: PyTorch backend -> Try download
        if backend == "pytorch":
            if not self.auto_download:
                raise FileNotFoundError(
                    f"Model not found: {weights_path}\n"
                    f"Use --auto-download to fetch it automatically."
                )
            return self._download_pt_model(path, model_type)

        # Case 2: ONNX/TensorRT backend -> Need to export from PT
        elif backend in ["onnx", "tensorrt"]:
            # Find corresponding PT weights
            pt_path = path.with_suffix(".pt")
            
            # If PT exists -> Export
            if pt_path.exists():
                if not self.auto_export:
                     raise FileNotFoundError(
                        f"ONNX/TensorRT model not found: {weights_path}\n"
                        f"PyTorch source found: {pt_path}\n"
                        f"Use --auto-export to generate it."
                    )
                return self._export_model(pt_path, path, backend)
            
            # If PT does not exist -> Download PT -> Export
            if self.auto_download and self.auto_export:
                print(f"Downloading source PT model to export...")
                if self._download_pt_model(pt_path, model_type):
                    return self._export_model(pt_path, path, backend)
            
            raise FileNotFoundError(
                f"Required model missing: {weights_path}\n"
                f"Source PT also missing: {pt_path}\n"
                f"Use --auto-download --auto-export to prepare automatically."
            )

        return False

    def _download_pt_model(self, target_path: Path, model_type: str) -> bool:
        """Download PyTorch model using Ultralytics API."""
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics library is required to download models.")

        try:
            print(f"📥 Downloading {target_path.name}...")
            model_name = target_path.stem  # e.g., yolov8n
            
            if "yolo" in model_type.lower():
                model_obj = YOLO(model_name)
            else:
                model_obj = RTDETR(model_name)

            # Ultralytics downloads to current dir, move if needed
            # Assuming it downloads to target_path.name directly
            downloaded = Path(model_name + ".pt")
            if downloaded.exists() and str(downloaded) != str(target_path):
                target_path.parent.mkdir(parents=True, exist_ok=True)
                downloaded.rename(target_path)
            
            print(f"✅ Model saved to: {target_path}")
            return True
        except Exception as e:
            print(f"❌ Download failed: {e}")
            raise

    def _export_model(self, source_pt: Path, target_path: Path, backend: str) -> bool:
        """Export model to ONNX or TensorRT."""
        try:
            print(f"🔄 Exporting {source_pt.name} to {backend.upper()}...")
            if backend == "onnx":
                ModelExporter.export_to_onnx(str(source_pt))
            else:
                ModelExporter.export_to_tensorrt(str(source_pt))
            
            print(f"✅ Export complete.")
            return True
        except Exception as e:
            print(f"❌ Export failed: {e}")
            raise