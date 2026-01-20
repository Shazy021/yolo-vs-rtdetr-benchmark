"""
Person Detection Video Processing Pipeline

Supports multiple models (YOLO, RT-DETR) and backends (PyTorch, ONNX, TensorRT).
Uses config.yaml for default settings, with CLI argument overrides.
"""

import argparse
import sys
from pathlib import Path

from src.detectors import ONNXDetector, RTDETRDetector, TRTDetector, YOLODetector
from src.export import ModelExporter
from src.utils import MetricsTracker, get_video_optimal_size, load_config, process_video


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Person detection in video with configurable backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config.yaml defaults
  python main.py --source video.mp4 --output result.mp4

  # Override model and backend
  python main.py --source video.mp4 --model yolo --backend onnx

  # Custom weights and thresholds
  python main.py --source video.mp4 --weights custom.onnx --conf 0.3

  # Preview mode with metrics
  python main.py --source video.mp4 --show --save-metrics metrics.json

  # Use custom config file
  python main.py --config myconfig.yaml --source video.mp4
        """,
    )

    # Required arguments
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video (default: output/result.mp4)")

    # Configuration
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (default: config.yaml)")

    # Model selection (overrides config)
    parser.add_argument(
        "--model", type=str, choices=["yolo", "rtdetr"], default=None, help="Model architecture (default: from config)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "onnx", "tensorrt"],
        default=None,
        help="Inference backend (default: from config)",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights (overrides config)")

    # Inference parameters (override config)
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold (default: from config)")
    parser.add_argument(
        "--nms", type=float, default=None, help="NMS threshold for ONNX/TensorRT (default: from config)"
    )

    # Video processing (override config)
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process (default: all)")
    parser.add_argument("--show", action="store_true", help="Show preview window during processing")
    parser.add_argument("--no-display-info", action="store_true", help="Hide FPS/detection overlay on video")

    # Metrics
    parser.add_argument("--save-metrics", type=str, default=None, help="Save metrics to JSON file")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics tracking")

    # Comparison mode
    parser.add_argument(
        "--comparison-mode",
        type=str,
        choices=["fair", "adaptive"],
        default=None,
        help="Input size strategy (default: from config)",
    )

    return parser.parse_args()


def get_model_info(args, config):
    """
    Determine model, backend, and weights path.

    Priority: CLI args > config file
    """
    # Get model and backend (CLI or config defaults)
    model = args.model or "yolo"
    backend = args.backend or "pytorch"

    # Get weights path
    if args.weights:
        weights_path = args.weights
    else:
        try:
            weights_path = config.get_model_path(model, backend)
        except KeyError as e:
            print(f"❌ Error: {e}")
            print(f"\nAvailable models in config:")
            for m in config.data["models"]:
                print(f"  {m}: {list(config.data['models'][m].keys())}")
            sys.exit(1)

    return model, backend, weights_path


def ensure_model_exists(weights_path: str, model: str, backend: str) -> bool:
    """
    Check if model exists, download and export if needed.

    Args:
        weights_path: Path to weights file
        model: Model name ('yolo' or 'rtdetr')
        backend: Backend type ('pytorch', 'onnx', 'tensorrt')

    Returns:
        True if model exists or was prepared successfully
    """
    weights_path_obj = Path(weights_path)

    # Model already exists
    if weights_path_obj.exists():
        return True

    print(f"\n⚠️  Model not found: {weights_path}")

    # For PyTorch backend, try auto-download
    if backend == "pytorch":
        if any(name in weights_path for name in ["yolov8", "yolo11", "rtdetr"]):
            response = input(f"Download {weights_path_obj.name} automatically? [y/n]: ")
            if response.lower() == "y":
                try:
                    from ultralytics import RTDETR, YOLO

                    print(f"\n📥 Downloading {weights_path_obj.name}...")

                    if "yolo" in model.lower():
                        model_obj = YOLO(weights_path_obj.name)
                    else:
                        model_obj = RTDETR(weights_path_obj.name)

                    # Move to correct location
                    downloaded_path = Path(weights_path_obj.name)
                    if downloaded_path.exists():
                        weights_path_obj.parent.mkdir(parents=True, exist_ok=True)
                        downloaded_path.rename(weights_path)
                        print(f"✅ Model saved to: {weights_path}")
                        return True
                except Exception as e:
                    print(f"❌ Download failed: {e}")
                    return False

    # For ONNX/TensorRT, offer to export from PyTorch
    elif backend in ["onnx", "tensorrt"]:
        # Find corresponding PyTorch weights
        pt_path = weights_path_obj.with_suffix(".pt")

        print(f"\n💡 {backend.upper()} model requires export from PyTorch weights.")
        print(f"   Looking for: {pt_path}")

        # Check if PT model exists
        if not pt_path.exists():
            response = input(f"\nDownload {pt_path.name} and export to {backend.upper()}? [y/n]: ")
            if response.lower() != "y":
                return False

            # Download PT model first
            try:
                from ultralytics import RTDETR, YOLO

                print(f"\n📥 Downloading {pt_path.name}...")

                if "yolo" in model.lower():
                    model_obj = YOLO(pt_path.name)
                else:
                    model_obj = RTDETR(pt_path.name)

                # Move to correct location
                downloaded_pt = Path(pt_path.name)
                if downloaded_pt.exists():
                    pt_path.parent.mkdir(parents=True, exist_ok=True)
                    downloaded_pt.rename(pt_path)
                    print(f"✅ PyTorch model saved to: {pt_path}")
            except Exception as e:
                print(f"❌ Download failed: {e}")
                return False
        else:
            response = input(f"\nExport {pt_path.name} to {backend.upper()}? [y/n]: ")
            if response.lower() != "y":
                return False

        # Export to requested format
        try:
            print(f"\n🔄 Exporting to {backend.upper()}...")

            if backend == "onnx":
                exported_path = ModelExporter.export_to_onnx(model_path=str(pt_path), opset=20, simplify=True)
            else:  # tensorrt
                exported_path = ModelExporter.export_to_tensorrt(model_path=str(pt_path), fp16=True, workspace=4)

            # Move to expected location if different
            exported_path_obj = Path(exported_path)
            if exported_path_obj != weights_path_obj:
                weights_path_obj.parent.mkdir(parents=True, exist_ok=True)
                exported_path_obj.rename(weights_path_obj)
                print(f"✅ Model moved to: {weights_path_obj}")

            return True

        except Exception as e:
            print(f"❌ Export failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    print("\nPlease provide the model manually or specify different weights with --weights")
    return False


def main():
    """Main execution function."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nCreate config.yaml first or specify with --config")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Config error: {e}")
        sys.exit(1)

    # Merge CLI arguments (CLI takes priority)
    config.merge_cli_args(args)

    # Print configuration summary
    config.print_summary()

    # Determine model, backend, and weights
    model, backend, weights_path = get_model_info(args, config)

    print(f"\n🎯 Selected configuration:")
    print(f"   Model: {model.upper()}")
    print(f"   Backend: {backend.upper()}")
    print(f"   Weights: {weights_path}")

    # Check if model exists (download/export if needed)
    if not ensure_model_exists(weights_path, model, backend):
        sys.exit(1)

    # Get inference parameters from config
    conf_threshold = config.get("inference.conf_threshold", 0.25)
    nms_threshold = config.get("inference.nms_threshold", 0.45)

    # Determine input size
    comparison_mode = args.comparison_mode or ("fair" if config.get("comparison.fair_mode") else "adaptive")

    if comparison_mode == "fair":
        optimal_size = tuple(config.get("comparison.reference_size", [640, 640]))
        print(f"\n📐 Fair comparison mode: {optimal_size[1]}x{optimal_size[0]}")
    else:
        try:
            video_info = get_video_optimal_size(args.source)
            optimal_size = (video_info["optimal_height"], video_info["optimal_width"])
            print(f"\n📐 Adaptive mode: {optimal_size[1]}x{optimal_size[0]}")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not analyze video: {e}")
            optimal_size = (640, 640)

    # Initialize detector
    print(f"\n🔧 Initializing {backend.upper()} detector...")
    try:
        if backend == "pytorch":
            if model == "yolo":
                detector = YOLODetector(weights_path, conf_threshold, optimal_size)
            else:
                detector = RTDETRDetector(weights_path, conf_threshold, optimal_size)

        elif backend == "onnx":
            detector = ONNXDetector(
                weights_path,
                use_gpu=config.get("inference.device.use_gpu", True),
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                model_type=model,
                input_size=optimal_size,
            )

        elif backend == "tensorrt":
            detector = TRTDetector(
                weights_path,
                model_type=model,
                conf_threshold=conf_threshold,
                img_size=optimal_size,
            )

    except Exception as e:
        print(f"❌ Error loading detector: {e}")
        print("\nTroubleshooting:")
        print("  - For ONNX: pip install onnxruntime-gpu")
        print("  - For TensorRT: ensure CUDA and TensorRT are installed")
        print("  - Check that weights file is valid")
        sys.exit(1)

    # Setup metrics tracker
    metrics_tracker = None
    if not args.no_metrics and config.get("metrics.enabled", True):
        metrics_name = f"{model}_{backend}"
        metrics_tracker = MetricsTracker(name=metrics_name)

    # Setup output path
    output_path = args.output
    if output_path is None:
        output_dir = Path(config.get("video.output.save_dir", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{Path(args.source).stem}_{model}_{backend}.mp4"
        output_path = str(output_dir / output_filename)

    # Process video
    print(f"\n🎬 Processing video...")
    try:
        stats = process_video(
            detector=detector,
            source_path=args.source,
            output_path=output_path,
            max_frames=config.get("video.max_frames"),
            show_preview=config.get("video.show_preview", False),
            metrics_tracker=metrics_tracker,
            display_info=config.get("video.display_info", True),
        )

        # Print and save metrics
        if metrics_tracker:
            metrics_tracker.print_summary()

            # Save to file if requested
            if args.save_metrics:
                metrics_tracker.save_to_file(args.save_metrics)
            elif config.get("metrics.save_to_file"):
                metrics_dir = Path(config.get("metrics.output_dir", "metrics"))
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = metrics_dir / f"{model}_{backend}_metrics.json"
                metrics_tracker.save_to_file(str(metrics_path))

        print(f"\n✅ SUCCESS! Output saved to: {output_path}")

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
