import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Person detection in video with configurable backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--source", type=str, required=True, help="Path to input video file")
    parser.add_argument("--output", type=str, default=None, help="Path to output video")

    # Configuration
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")

    # Model selection
    parser.add_argument(
        "--model", type=str, choices=["yolo", "rtdetr"], default=None, help="Model architecture"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["pytorch", "onnx", "tensorrt"],
        default=None,
        help="Inference backend",
    )
    parser.add_argument("--weights", type=str, default=None, help="Path to model weights")

    # Inference parameters
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=None, help="NMS threshold")

    # Processing
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--show", action="store_true", help="Show preview window")
    parser.add_argument("--no-display-info", action="store_true", help="Hide overlay info")

    # Metrics
    parser.add_argument("--save-metrics", type=str, default=None, help="Save metrics to JSON")
    parser.add_argument("--no-metrics", action="store_true", help="Disable metrics")

    # Comparison mode
    parser.add_argument(
        "--comparison-mode",
        type=str,
        choices=["fair", "adaptive"],
        default=None,
        help="Input size strategy",
    )

    return parser.parse_args()