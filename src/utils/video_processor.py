from typing import Optional, Dict, Any, Tuple
import cv2
import time
import numpy as np
from pathlib import Path


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
        'original_width': original_width,
        'original_height': original_height,
        'optimal_width': optimal_w,
        'optimal_height': optimal_h,
        'fps': fps,
        'total_frames': total_frames
    }


class VideoProcessor:
    """
    Enhanced video processor with beautiful visualization and compression.

    Features:
    - Optimized H.264 encoding for smaller file sizes
    - Semi-transparent overlay panels
    - Color-coded bounding boxes by confidence
    - Real-time performance metrics display
    """

    # Color scheme for confidence levels
    CONFIDENCE_COLORS = {
        'high': (0, 255, 0),      # Green (conf >= 0.7)
        'medium': (0, 165, 255),  # Orange (0.5 <= conf < 0.7)
        'low': (0, 100, 255)      # Red (conf < 0.5)
    }

    @staticmethod
    def get_best_codec() -> Tuple[str, int]:
        """
        Get best available video codec for current platform.

        Tries H.264 variants first (best compression), falls back to MPEG-4.
        Performs actual test write to verify codec availability.

        Returns:
            Tuple of (codec_name, fourcc_code)
        """
        # Try H.264 codecs (best compression)
        codecs_to_try = [
            ('H264', cv2.VideoWriter_fourcc(*'H264')),
            ('X264', cv2.VideoWriter_fourcc(*'X264')),
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4 (fallback)
        ]

        # Test each codec
        for codec_name, fourcc in codecs_to_try:
            test_writer = cv2.VideoWriter(
                'test_codec.mp4',
                fourcc,
                30,
                (640, 480)
            )
            if test_writer.isOpened():
                test_writer.release()
                Path('test_codec.mp4').unlink(missing_ok=True)
                print(f"   Using codec: {codec_name}")
                return codec_name, fourcc

        # Fallback to mp4v
        print("   Warning: Using fallback codec mp4v")
        return 'mp4v', cv2.VideoWriter_fourcc(*'mp4v')

    @staticmethod
    def draw_beautiful_bbox(
        frame: np.ndarray,
        bbox: list,
        conf: float,
        label: str = "Person",
        thickness: int = 1
    ) -> np.ndarray:
        """
        Draw beautiful bounding box with color-coded confidence.

        Features:
        - Corner accents for modern look
        - Color-coded by confidence level
        - Semi-transparent label background
        - Anti-aliased text

        Args:
            frame: Input frame
            bbox: [x1, y1, x2, y2] coordinates
            conf: Confidence score (0.0-1.0)
            label: Object label text
            thickness: Box line thickness

        Returns:
            Frame with drawn bounding box
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Choose color based on confidence
        if conf >= 0.7:
            color = VideoProcessor.CONFIDENCE_COLORS['high']
        elif conf >= 0.5:
            color = VideoProcessor.CONFIDENCE_COLORS['medium']
        else:
            color = VideoProcessor.CONFIDENCE_COLORS['low']

        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw corner accents (stylish effect)
        corner_length = min(20, (x2 - x1) // 5, (y2 - y1) // 2)
        accent_thickness = thickness

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, accent_thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, accent_thickness)

        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, accent_thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, accent_thickness)

        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, accent_thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, accent_thickness)

        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, accent_thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, accent_thickness)

        # Prepare label text
        label_text = f"{label} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1

        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )

        # Draw label background (semi-transparent)
        label_bg_y1 = max(0, y1 - text_h - 10)
        label_bg_y2 = y1

        # Create overlay for transparency
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x1, label_bg_y1),
            (x1 + text_w + 10, label_bg_y2),
            color,
            -1
        )

        # Blend overlay (transparency effect)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw label text
        cv2.putText(
            frame,
            label_text,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            font_thickness,
            cv2.LINE_AA
        )

        return frame

    @staticmethod
    def draw_stats_panel(
        frame: np.ndarray,
        fps: float,
        num_detections: int,
        frame_num: int,
        total_frames: int
    ) -> np.ndarray:
        """
        Draw beautiful semi-transparent statistics panel.

        Panel displays:
        - Current FPS (color-coded by performance)
        - Number of detected people
        - Processing progress percentage

        Args:
            frame: Input frame
            fps: Current FPS
            num_detections: Number of detections
            frame_num: Current frame number
            total_frames: Total frames in video

        Returns:
            Frame with stats panel overlay
        """
        # Panel dimensions
        panel_height = 100
        panel_width = 350
        panel_x = 10
        panel_y = 10

        # Create semi-transparent overlay
        overlay = frame.copy()

        # Draw panel background
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (40, 40, 40),  # Dark gray
            -1
        )

        # Blend for transparency
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw panel border
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_width, panel_y + panel_height),
            (100, 100, 100),  # Light gray border
            2
        )

        # Prepare text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_spacing = 30
        text_x = panel_x + 15
        text_y = panel_y + 30

        # FPS (color-coded by performance)
        fps_color = (0, 255, 0) if fps >= 30 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (text_x, text_y),
            font,
            font_scale,
            fps_color,
            font_thickness,
            cv2.LINE_AA
        )

        # Detections count
        cv2.putText(
            frame,
            f"People: {num_detections}",
            (text_x, text_y + line_spacing),
            font,
            font_scale,
            (255, 255, 255),  # White
            font_thickness,
            cv2.LINE_AA
        )

        # Progress
        progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(
            frame,
            f"Progress: {progress:.1f}%",
            (text_x, text_y + 2 * line_spacing),
            font,
            font_scale,
            (255, 255, 0),  # Cyan
            font_thickness,
            cv2.LINE_AA
        )

        return frame


def process_video(
    detector,
    source_path: str,
    output_path: str,
    max_frames: Optional[int] = None,
    show_preview: bool = False,
    metrics_tracker=None,
    display_info: bool = True
) -> Dict[str, Any]:
    """
    Process video file with beautiful person detection visualization.

    This function provides production-quality video processing with:
    - Optimized H.264 encoding for file size reduction
    - Beautiful color-coded bounding boxes
    - Semi-transparent statistics overlay
    - Real-time performance tracking

    Args:
        detector: Instance of BaseDetector
        source_path: Path to input video file
        output_path: Path to output video file
        max_frames: Maximum frames to process (None = all)
        show_preview: Display preview window during processing
        metrics_tracker: Optional MetricsTracker instance
        display_info: Show statistics overlay on frames

    Returns:
        Dictionary with processing statistics:
            - frame_count: Frames processed
            - total_time: Processing time (seconds)
            - avg_fps: Average processing FPS
            - output_path: Path to output file
            - input_size_mb: Input file size
            - output_size_mb: Output file size
            - compression_ratio: Compression percentage

    Raises:
        ValueError: If video cannot be opened or created
    """
    # Open input video
    print(source_path)
    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {source_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    input_size_mb = Path(source_path).stat().st_size / (1024 * 1024)

    print(f"📹 Input video info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   File size: {input_size_mb:.2f} MB")

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Get best available codec for compression
    codec_name, fourcc = VideoProcessor.get_best_codec()

    # Initialize video writer with optimized settings
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        raise ValueError(f"Cannot create output video: {output_path}")

    # Processing loop
    frame_count = 0
    start_time = time.time()

    print(f"🚀 Processing video with {codec_name} codec...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference
            inference_start = time.time()
            detections = detector.predict(frame)
            inference_time = (time.time() - inference_start) * 1000  # ms

            # Draw beautiful bounding boxes
            annotated_frame = frame.copy()
            for det in detections:
                annotated_frame = VideoProcessor.draw_beautiful_bbox(
                    annotated_frame,
                    det['bbox'],
                    det['conf'],
                    label="Person"
                )

            # Add statistics panel
            if display_info:
                current_fps = 1000 / inference_time if inference_time > 0 else 0
                annotated_frame = VideoProcessor.draw_stats_panel(
                    annotated_frame,
                    current_fps,
                    len(detections),
                    frame_count + 1,
                    total_frames if total_frames > 0 else max_frames or 1
                )

            # Write frame
            out.write(annotated_frame)

            # Collect metrics
            if metrics_tracker:
                metrics_tracker.add_frame(inference_time, len(detections))

            # Preview
            if show_preview:
                cv2.imshow('Detection Preview', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("   Preview interrupted by user")
                    break

            frame_count += 1

            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                current_fps = 1000 / inference_time if inference_time > 0 else 0
                print(f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                      f"{current_fps:.1f} FPS - {len(detections)} people")

            # Stop at max_frames
            if max_frames and frame_count >= max_frames:
                print(f"   Reached max_frames limit: {max_frames}")
                break

    finally:
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()

    # Calculate stats
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    # Get output file size
    output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    compression_ratio = (1 - output_size_mb / input_size_mb) * 100 if input_size_mb > 0 else 0

    stats = {
        'frame_count': frame_count,
        'total_time': total_time,
        'avg_fps': avg_fps,
        'output_path': output_path,
        'input_size_mb': input_size_mb,
        'output_size_mb': output_size_mb,
        'compression_ratio': compression_ratio
    }

    print(f"\n✅ Processing complete!")
    print(f"   Frames processed: {frame_count}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average FPS: {avg_fps:.2f}")
    print(f"   Output size: {output_size_mb:.2f} MB")
    if compression_ratio > 0:
        print(f"   Compression: {abs(compression_ratio):.1f}% {'smaller' if compression_ratio > 0 else 'larger'}")
    else:
        print(f"   Size change: {abs(compression_ratio):.1f}% larger")
    print(f"   Saved: {output_path}")

    return stats