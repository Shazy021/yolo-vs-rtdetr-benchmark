import logging
import sys
from typing import Any, Dict, Optional, Tuple

import cv2


def setup_logging(verbose: bool = False):
    """
    Configure logging format and level.

    Args:
        verbose: If True, set level to DEBUG. Otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create a custom formatter to include colors in console (if supported)
    # Simple clean format: [LEVEL] Message
    log_format = "[%(levelname)s] %(message)s"
    
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Suppress verbose logs from external libraries if needed
    logging.getLogger("ultralytics").setLevel(logging.WARNING)


def get_optimal_size(width: int, height: int, max_size: int = None) -> Tuple[int, int]:
    """
    Calculate optimal input size for inference (multiple of 32).

    Most detection models require input dimensions divisible by 32
    for proper feature map computation.

    Args:
        width: Original video width
        height: Original video height
        max_size: Maximum dimension limit (optional, prevents upscaling)

    Returns:
        Tuple of (height, width) rounded to nearest multiple of 32,
        minimum 320 pixels per dimension
    """
    # Apply max_size constraint if specified (no upscaling)
    if max_size is not None:
        scale = min(max_size / width, max_size / height, 1.0)
        width = int(width * scale)
        height = int(height * scale)

    # Round to nearest multiple of 32
    optimal_w = (width // 32) * 32
    optimal_h = (height // 32) * 32

    # Enforce minimum size of 320
    optimal_w = max(320, optimal_w)
    optimal_h = max(320, optimal_h)

    return optimal_h, optimal_w


def get_video_optimal_size(video_path: str, max_size: int = None) -> Dict[str, Any]:
    """
    Analyze video and calculate optimal inference dimensions.

    Reads video metadata and computes model-friendly dimensions
    (multiples of 32) while preserving aspect ratio.

    Args:
        video_path: Path to video file
        max_size: Maximum dimension for inference (optional)

    Returns:
        Dictionary containing:
            - original_width: Original video width
            - original_height: Original video height
            - optimal_width: Computed optimal width (multiple of 32)
            - optimal_height: Computed optimal height (multiple of 32)
            - fps: Video frames per second
            - total_frames: Total frame count

    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    # Calculate optimal size
    optimal_h, optimal_w = get_optimal_size(original_width, original_height, max_size)

    return {
        "original_width": original_width,
        "original_height": original_height,
        "optimal_width": optimal_w,
        "optimal_height": optimal_h,
        "fps": fps,
        "total_frames": total_frames,
    }