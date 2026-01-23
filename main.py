"""
Person Detection Video Processing Pipeline

Supports multiple models (YOLO, RT-DETR) and backends (PyTorch, ONNX, TensorRT).
Uses config.yaml for default settings, with CLI argument overrides.
"""

import sys
from pathlib import Path

from src.detectors import DetectorFactory
from src.utils import MetricsTracker, ModelManager, get_video_optimal_size, load_config, parse_args, process_video


def main():
    """Main execution function."""
    # 1. Parse CLI Arguments
    args = parse_args()

    # 2. Load Configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Config error: {e}")
        sys.exit(1)

    config.merge_cli_args(args)
    config.print_summary()

    # 3. Determine Model Paths
    # Use CLI args or fallback to config defaults
    model = args.model or config.get("models.default_model", "yolo")
    backend = args.backend or config.get("inference.default_backend", "pytorch")

    if args.weights:
        weights_path = args.weights
    else:
        try:
            weights_path = config.get_model_path(model, backend)
        except KeyError as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    # 4. Ensure Model Exists (Automation Logic)
    manager = ModelManager(config)

    try:
        manager.ensure_model(weights_path, model, backend)
    except Exception as e:
        print(f"❌ Model preparation failed: {e}")
        sys.exit(1)

    # 5. Determine Input Size
    comparison_mode = args.comparison_mode or ("fair" if config.get("comparison.fair_mode") else "adaptive")

    if comparison_mode == "fair":
        optimal_size = tuple(config.get("comparison.reference_size", [640, 640]))
    else:
        try:
            video_info = get_video_optimal_size(args.source)
            optimal_size = (video_info["optimal_height"], video_info["optimal_width"])
        except Exception as e:
            optimal_size = (640, 640)

    # 6. Create Detector via Factory
    conf_threshold = config.get("inference.conf_threshold", 0.25)
    nms_threshold = config.get("inference.nms_threshold", 0.45)
    use_gpu = config.get("inference.device.use_gpu", True)

    print(f"\n🔧 Initializing {backend.upper()} detector...")

    try:
        detector = DetectorFactory.create(
            model=model,
            backend=backend,
            weights_path=weights_path,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            input_size=optimal_size,
            use_gpu=use_gpu,
        )
    except Exception as e:
        print(f"❌ Failed to create detector: {e}")
        sys.exit(1)

    # 7. Setup Metrics
    metrics_tracker = None
    if not args.no_metrics and config.get("metrics.enabled", True):
        metrics_tracker = MetricsTracker(name=f"{model}_{backend}")

    # 8. Setup Output
    output_path = args.output
    if output_path is None:
        output_dir = Path(config.get("video.output.save_dir", "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{Path(args.source).stem}_{model}_{backend}.mp4"
        output_path = str(output_dir / output_filename)

    # 9. Process Video
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

        if metrics_tracker:
            metrics_tracker.print_summary()
            if args.save_metrics:
                metrics_tracker.save_to_file(args.save_metrics)
            elif config.get("metrics.save_to_file"):
                metrics_dir = Path(config.get("metrics.output_dir", "metrics"))
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = metrics_dir / f"{model}_{backend}_metrics.json"
                metrics_tracker.save_to_file(str(metrics_path))

        print(f"\n✅ SUCCESS! Output saved to: {output_path}")

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
