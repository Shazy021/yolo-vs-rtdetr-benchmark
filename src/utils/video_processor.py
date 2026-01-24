import threading
import time
from pathlib import Path
from queue import Queue
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


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


class VideoProcessor:
    """
    Enhanced video processor with beautiful visualization and compression.

    Features:
    - Optimized H.264 encoding for smaller file sizes
    - Semi-transparent overlay panels
    - Color-coded bounding boxes by confidence
    - Real-time performance metrics display
    """

    DEFAULT_COLORS = {
        "high": (0, 255, 0),  # Green
        "medium": (0, 165, 255),  # Orange
        "low": (0, 100, 255),  # Red
        "panel_bg": (40, 40, 40),  # Dark grey
        "panel_border": (100, 100, 100),  # Light gray
        "text_white": (255, 255, 255),
        "text_cyan": (255, 255, 0),
    }

    def __init__(
        self,
        colors: Dict[str, Tuple[int, int, int]] = None,
        panel_width: int = 350,
        panel_height: int = 100,
        corner_length_ratio: float = 0.05,  # FIXED: was 'conner_lenth_ratio'
    ):
        """
        Initialize Video Processor with visual settings.

        Args:
            colors: Dict overriding default colors.
            panel_width: Width of stats panel.
            panel_height: Height of stats panel.
            corner_length_ratio: Ratio of bbox side for accent corners.
        """
        self.colors = {**self.DEFAULT_COLORS, **(colors or {})}
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.corner_length_ratio = corner_length_ratio

    def create_video_writer(self, output_path: str, fps: int, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
        """
        Create video writer with best available codec.

        Args:
            output_path: Path to output video file
            fps: Frames per second
            frame_size: (width, height)

        Returns:
            Initialized VideoWriter object

        Raises:
            RuntimeError: If no codec works
        """
        fourcc = None

        # Priority list
        codecs_to_try = [
            ("H264", cv2.VideoWriter_fourcc(*"H264")),
            ("X264", cv2.VideoWriter_fourcc(*"X264")),
            ("mp4v", cv2.VideoWriter_fourcc(*"mp4v")),  # MPEG-4 (fallback)
        ]

        # Test each codec
        for codec_name, codec_fourcc in codecs_to_try:
            try:
                writer = cv2.VideoWriter(output_path, codec_fourcc, fps, frame_size)
                if writer.isOpened():
                    print(f"   Using codec: {codec_name}")
                    return writer
            except Exception:
                # Some backends raise exceptions immediately, some don't
                pass

        raise RuntimeError("Failed to initialize any video codec.")

    def draw_beautiful_bbox(
        self, frame: np.ndarray, bbox: list, conf: float, label: str = "Person", thickness: int = 1
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
            color = self.colors["high"]
        elif conf >= 0.5:
            color = self.colors["medium"]
        else:
            color = self.colors["low"]

        # Draw main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Draw corner accents (stylish effect)
        # Calculate dynamic corner length based on box size
        box_w = x2 - x1
        box_h = y2 - y1
        # FIXED: 'corner_lenght' -> 'corner_length'
        corner_length = min(20, int(box_w * self.corner_length_ratio), int(box_h * self.corner_length_ratio))

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
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # Draw label background (semi-transparent)
        label_bg_y1 = max(0, y1 - text_h - 10)
        label_bg_y2 = y1

        # Overlay logic (semi-transparent)
        # Instead of full frame copy, we only manipulate ROI to save memory
        roi_x1 = x1
        roi_y1 = label_bg_y1
        roi_x2 = x1 + text_w + 10
        roi_y2 = label_bg_y2

        # Clip ROI to frame bounds
        roi_x1 = max(0, roi_x1)
        roi_y1 = max(0, roi_y1)
        roi_x2 = min(frame.shape[1], roi_x2)
        roi_y2 = min(frame.shape[0], roi_y2)

        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            overlay = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()
            cv2.rectangle(overlay, (0, 0), (roi_x2 - roi_x1, roi_y2 - roi_y1), color, -1)

            # Alpha blend only ROI
            alpha = 0.7
            frame[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.addWeighted(
                overlay, alpha, frame[roi_y1:roi_y2, roi_x1:roi_x2], 1 - alpha, 0
            )

            cv2.putText(
                frame,
                label_text,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                self.colors["text_white"],
                font_thickness,
                cv2.LINE_AA,
            )

        return frame

    def draw_stats_panel(
        self, frame: np.ndarray, fps: float, num_detections: int, frame_num: int, total_frames: int
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
        panel_x = 10
        panel_y = 10

        # Define ROI for panel
        roi_x1 = panel_x
        roi_y1 = panel_y
        roi_x2 = panel_x + self.panel_width
        roi_y2 = panel_y + self.panel_height

        # Clip
        roi_x2 = min(frame.shape[1], roi_x2)
        roi_y2 = min(frame.shape[0], roi_y2)

        overlay = frame[roi_y1:roi_y2, roi_x1:roi_x2].copy()

        # Background
        cv2.rectangle(overlay, (0, 0), (roi_x2 - roi_x1, roi_y2 - roi_y1), self.colors["panel_bg"], -1)

        # Blend
        alpha = 0.6
        frame[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.addWeighted(
            overlay, alpha, frame[roi_y1:roi_y2, roi_x1:roi_x2], 1 - alpha, 0
        )

        # Border
        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), self.colors["panel_border"], 2)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        line_spacing = 30
        text_x = panel_x + 15
        text_y = panel_y + 30

        # FPS
        fps_color = (0, 255, 0) if fps >= 30 else (0, 165, 255) if fps >= 15 else (0, 0, 255)
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (text_x, text_y), font, font_scale, fps_color, font_thickness, cv2.LINE_AA
        )

        # Detections
        cv2.putText(
            frame,
            f"People: {num_detections}",
            (text_x, text_y + line_spacing),
            font,
            font_scale,
            self.colors["text_white"],
            font_thickness,
            cv2.LINE_AA,
        )

        # Progress
        progress = (frame_num / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(
            frame,
            f"Progress: {progress:.1f}%",
            (text_x, text_y + 2 * line_spacing),
            font,
            font_scale,
            self.colors["text_cyan"],
            font_thickness,
            cv2.LINE_AA,
        )

        return frame


def process_video(
    detector,
    source_path: str,
    output_path: str,
    max_frames: Optional[int] = None,
    show_preview: bool = False,
    metrics_tracker=None,
    display_info: bool = True,
    viz_config: Dict[str, Any] = None,
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
        viz_config: Optional dict to customize visualization colors/sizes (passed to VideoProcessor).

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
    # Initialize Processor with config
    processor = VideoProcessor(colors=viz_config.get("colors") if viz_config else None)

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

    # Setup Threaded Writer
    frame_queue = Queue(maxsize=10)

    # Flag to capture errors from the writer thread
    writer_error = [None]

    def _writer_worker():
        """Worker function running in a separate thread for writing video."""
        try:
            # Initialize writer INSIDE the thread (critical for OpenCV stability)
            out = processor.create_video_writer(output_path, fps, (width, height))

            while True:
                # Wait for frame from queue. .get() blocks until an item is available.
                frame = frame_queue.get()

                # If None is received, it's the stop signal
                if frame is None:
                    break

                # Write frame to disk
                out.write(frame)
                # Notify the queue that the task is done
                frame_queue.task_done()

            out.release()

        except Exception as e:
            writer_error[0] = e

    # Start the writer thread
    writer_thread = threading.Thread(target=_writer_worker, daemon=True)
    writer_thread.start()

    # Main Processing Loop
    frame_count = 0
    start_time = time.time()
    print(f"🚀 Processing video with threaded I/O...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run Inference (Main Thread)
            inference_start = time.time()
            detections = detector.predict(frame)
            inference_time = (time.time() - inference_start) * 1000  # ms

            # Draw Visualization
            annotated_frame = frame
            for det in detections:
                annotated_frame = processor.draw_beautiful_bbox(
                    annotated_frame, det["bbox"], det["conf"], label="Person"
                )

            if display_info:
                current_fps = 1000 / inference_time if inference_time > 0 else 0
                annotated_frame = processor.draw_stats_panel(
                    annotated_frame,
                    current_fps,
                    len(detections),
                    frame_count + 1,
                    total_frames if total_frames > 0 else max_frames or 1,
                )

            # Push to Queue (Threaded I/O)
            frame_queue.put(annotated_frame)

            # Collect metrics
            if metrics_tracker:
                metrics_tracker.add_frame(inference_time, len(detections))

            if show_preview:
                cv2.imshow("Detection Preview", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("   Preview interrupted by user")
                    break

            frame_count += 1

            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                current_fps = 1000 / inference_time if inference_time > 0 else 0
                print(
                    f"   Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                    f"{current_fps:.1f} FPS - {len(detections)} people"
                )

            if max_frames and frame_count >= max_frames:
                print(f"   Reached max_frames limit: {max_frames}")
                break

    finally:
        cap.release()

        # Signal the writer thread to stop
        frame_queue.put(None)

        # Wait for the writer thread to finish (with timeout)
        writer_thread.join(timeout=5.0)

        if show_preview:
            cv2.destroyAllWindows()

        # Check for errors raised in the writer thread
        if writer_error[0] is not None:
            raise RuntimeError(f"Video Writer thread failed: {writer_error[0]}")

    # Final Statistics
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0

    output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    compression_ratio = (1 - output_size_mb / input_size_mb) * 100 if input_size_mb > 0 else 0

    stats = {
        "frame_count": frame_count,
        "total_time": total_time,
        "avg_fps": avg_fps,
        "output_path": output_path,
        "input_size_mb": input_size_mb,
        "output_size_mb": output_size_mb,
        "compression_ratio": compression_ratio,
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
