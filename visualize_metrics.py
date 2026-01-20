import argparse
import glob as glob_module
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    import numpy as np
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("❌ Error: matplotlib is required for visualization")
    print("   Install with: pip install matplotlib")
    sys.exit(1)


def remove_outliers_iqr(data: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, int]:
    """
    Remove outliers using IQR method.

    Args:
        data: Input data array
        factor: IQR multiplier (1.5 = standard, 3.0 = conservative)

    Returns:
        Tuple of (filtered_data, num_outliers_removed)
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr

    mask = (data >= lower_bound) & (data <= upper_bound)
    filtered = data[mask]
    outliers_removed = len(data) - len(filtered)

    return filtered, outliers_removed


class MetricsVisualizer:
    """Enhanced visualizer with warmup handling."""

    def __init__(self, output_dir: str = "plots", skip_warmup: int = 0, remove_outliers: bool = True):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
            skip_warmup: Number of initial frames to skip (warmup period)
            remove_outliers: Remove statistical outliers from distributions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_warmup = skip_warmup
        self.remove_outliers = remove_outliers

        plt.style.use("seaborn-v0_8-darkgrid")
        self.colors = plt.cm.Set2(np.linspace(0, 1, 8))

    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """Load metrics from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        # Apply warmup skip
        if self.skip_warmup > 0 and "frames" in data:
            original_count = len(data["frames"])
            data["frames"] = data["frames"][self.skip_warmup :]
            if data["frames"]:
                print(f"   Skipped {self.skip_warmup} warmup frames ({original_count} → {len(data['frames'])})")

        return data

    def plot_latency_over_time(self, metrics_list: List[Dict], output_name: str = "latency_over_time.png"):
        """Plot latency over time."""
        fig, ax = plt.subplots(figsize=(14, 6))

        for idx, metrics in enumerate(metrics_list):
            name = metrics["name"]
            frames = metrics.get("frames", [])

            if not frames:
                continue

            frame_nums = [f["frame_num"] for f in frames]
            latencies = [f["latency_ms"] for f in frames]

            ax.plot(
                frame_nums, latencies, label=name, color=self.colors[idx % len(self.colors)], alpha=0.7, linewidth=1.5
            )

        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=12)

        title = "Inference Latency Over Time"
        if self.skip_warmup > 0:
            title += f" (warmup {self.skip_warmup} frames excluded)"
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {output_path}")

    def plot_fps_over_time(self, metrics_list: List[Dict], output_name: str = "fps_over_time.png"):
        """Plot FPS over time."""
        fig, ax = plt.subplots(figsize=(14, 6))

        for idx, metrics in enumerate(metrics_list):
            name = metrics["name"]
            frames = metrics.get("frames", [])

            if not frames:
                continue

            frame_nums = [f["frame_num"] for f in frames]
            fps_values = [f["fps"] for f in frames]

            ax.plot(
                frame_nums, fps_values, label=name, color=self.colors[idx % len(self.colors)], alpha=0.7, linewidth=1.5
            )

        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("FPS", fontsize=12)

        title = "Processing FPS Over Time"
        if self.skip_warmup > 0:
            title += f" (warmup {self.skip_warmup} frames excluded)"
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        ax.axhline(y=30, color="green", linestyle="--", alpha=0.5, linewidth=1)
        ax.axhline(y=60, color="blue", linestyle="--", alpha=0.5, linewidth=1)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {output_path}")

    def plot_latency_distribution(self, metrics_list: List[Dict], output_name: str = "latency_distribution.png"):
        """
        Plot latency distribution with grid layout (max 3 cols).
        """
        n_metrics = len(metrics_list)
        if n_metrics == 0:
            return

        # Calculate grid dimensions (max 3 columns)
        cols = min(n_metrics, 3)
        rows = n_metrics // cols

        # Dynamic figure size: Width based on cols, Height based on rows
        figsize = (6 * cols, 5 * rows)
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Flatten axes array for easy iteration
        if n_metrics > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for idx, ax in enumerate(axes_flat):
            # If we have more slots than metrics (e.g. 5 metrics in 2x3 grid), hide the extra
            if idx >= n_metrics:
                ax.axis("off")
                continue

            metrics = metrics_list[idx]
            name = metrics["name"]
            frames = metrics.get("frames", [])

            if not frames:
                continue

            latencies_raw = np.array([f["latency_ms"] for f in frames])

            # Remove outliers
            if self.remove_outliers and len(latencies_raw) > 10:
                latencies, outliers_removed = remove_outliers_iqr(latencies_raw, factor=1.5)
                outlier_pct = (outliers_removed / len(latencies_raw)) * 100
            else:
                latencies = latencies_raw
                outliers_removed = 0
                outlier_pct = 0

            # Statistics
            mean_lat = np.mean(latencies)
            median_lat = np.median(latencies)
            p95 = np.percentile(latencies, 95)

            # Bin range
            bin_min = max(0, np.percentile(latencies, 0.5))
            bin_max = np.percentile(latencies, 99.5)

            # Histogram
            ax.hist(
                latencies,
                bins=50,
                range=(bin_min, bin_max),
                color=self.colors[idx % len(self.colors)],
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            # Stats lines
            ax.axvline(mean_lat, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_lat:.1f}ms", zorder=10)
            ax.axvline(
                median_lat, color="green", linestyle="--", linewidth=2, label=f"Median: {median_lat:.1f}ms", zorder=10
            )
            ax.axvline(p95, color="orange", linestyle=":", linewidth=2, label=f"P95: {p95:.1f}ms", alpha=0.8, zorder=10)

            ax.set_xlabel("Latency (ms)", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)

            # Title
            title = f"{name}\nLatency Distribution"
            if self.skip_warmup > 0:
                title += f" (skip {self.skip_warmup})"
            if outliers_removed > 0:
                title += f"\n{outliers_removed} outliers removed ({outlier_pct:.1f}%)"

            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.3, axis="y")

            # Stats text
            stats_text = f"n={len(latencies)}\nσ={np.std(latencies):.1f}ms"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
            )

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {output_path}")

    def plot_percentile_comparison(self, metrics_list: List[Dict], output_name: str = "percentile_comparison.png"):
        """Plot percentile comparison."""
        fig, ax = plt.subplots(figsize=(12, 6))

        percentiles = ["min", "p50", "p95", "p99", "max"]
        x = np.arange(len(percentiles))
        width = 0.8 / len(metrics_list)

        for idx, metrics in enumerate(metrics_list):
            name = metrics["name"]
            summary = metrics.get("summary", {})

            if not summary:
                continue

            values = [
                summary.get("min_latency_ms", 0),
                summary.get("median_latency_ms", 0),
                summary.get("p95_latency_ms", 0),
                summary.get("p99_latency_ms", 0),
                summary.get("max_latency_ms", 0),
            ]

            offset = (idx - len(metrics_list) / 2) * width + width / 2
            bars = ax.bar(x + offset, values, width, label=name, color=self.colors[idx % len(self.colors)], alpha=0.8)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Percentile", fontsize=12)
        ax.set_ylabel("Latency (ms)", fontsize=12)

        title = "Latency Percentile Comparison"
        if self.skip_warmup > 0:
            title += f" (warmup {self.skip_warmup} frames excluded)"
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(["Min", "P50 (Median)", "P95", "P99", "Max"])
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {output_path}")

    def plot_summary_comparison(self, metrics_list: List[Dict], output_name: str = "summary_comparison.png"):
        """Plot comprehensive summary."""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        names = [m["name"] for m in metrics_list]
        x = np.arange(len(names))

        # 1. Average Latency
        ax1 = fig.add_subplot(gs[0, 0])
        avg_latencies = [m["summary"].get("avg_latency_ms", 0) for m in metrics_list]
        bars1 = ax1.bar(x, avg_latencies, color=self.colors[: len(names)], alpha=0.8)
        ax1.set_ylabel("Latency (ms)", fontsize=10)
        ax1.set_title("Average Latency", fontsize=11, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax1.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars1, avg_latencies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 2. Average FPS
        ax2 = fig.add_subplot(gs[0, 1])
        avg_fps = [m["summary"].get("avg_fps", 0) for m in metrics_list]
        bars2 = ax2.bar(x, avg_fps, color=self.colors[: len(names)], alpha=0.8)
        ax2.set_ylabel("FPS", fontsize=10)
        ax2.set_title("Average FPS", fontsize=11, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax2.axhline(y=30, color="green", linestyle="--", alpha=0.5)
        ax2.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars2, avg_fps):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 3. Standard Deviation
        ax3 = fig.add_subplot(gs[0, 2])
        std_devs = [m["summary"].get("std_latency_ms", 0) for m in metrics_list]
        bars3 = ax3.bar(x, std_devs, color=self.colors[: len(names)], alpha=0.8)
        ax3.set_ylabel("Std Dev (ms)", fontsize=10)
        ax3.set_title("Latency Std Deviation", fontsize=11, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax3.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars3, std_devs):
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 4. P95 Latency
        ax4 = fig.add_subplot(gs[1, 0])
        p95_vals = [m["summary"].get("p95_latency_ms", 0) for m in metrics_list]
        bars4 = ax4.bar(x, p95_vals, color=self.colors[: len(names)], alpha=0.8)
        ax4.set_ylabel("Latency (ms)", fontsize=10)
        ax4.set_title("P95 Latency", fontsize=11, fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax4.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars4, p95_vals):
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 5. P99 Latency
        ax5 = fig.add_subplot(gs[1, 1])
        p99_vals = [m["summary"].get("p99_latency_ms", 0) for m in metrics_list]
        bars5 = ax5.bar(x, p99_vals, color=self.colors[: len(names)], alpha=0.8)
        ax5.set_ylabel("Latency (ms)", fontsize=10)
        ax5.set_title("P99 Latency", fontsize=11, fontweight="bold")
        ax5.set_xticks(x)
        ax5.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax5.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars5, p99_vals):
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 6. Total Detections
        ax6 = fig.add_subplot(gs[1, 2])
        total_dets = [m["summary"].get("total_detections", 0) for m in metrics_list]
        bars6 = ax6.bar(x, total_dets, color=self.colors[: len(names)], alpha=0.8)
        ax6.set_ylabel("Count", fontsize=10)
        ax6.set_title("Total Detections", fontsize=11, fontweight="bold")
        ax6.set_xticks(x)
        ax6.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
        ax6.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars6, total_dets):
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{int(val)}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        title = "Performance Metrics Comparison"
        if self.skip_warmup > 0:
            title += f" (warmup {self.skip_warmup} frames excluded)"
        fig.suptitle(title, fontsize=16, fontweight="bold", y=0.995)

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {output_path}")

    def plot_detection_over_time(self, metrics_list: List[Dict], output_name: str = "detections_over_time.png"):
        """Plot detection count over time."""
        fig, ax = plt.subplots(figsize=(14, 6))

        for idx, metrics in enumerate(metrics_list):
            name = metrics["name"]
            frames = metrics.get("frames", [])

            if not frames:
                continue

            frame_nums = [f["frame_num"] for f in frames]
            detections = [f["detections"] for f in frames]

            ax.plot(
                frame_nums, detections, label=name, color=self.colors[idx % len(self.colors)], alpha=0.7, linewidth=1.5
            )

        ax.set_xlabel("Frame Number", fontsize=12)
        ax.set_ylabel("Number of Detections", fontsize=12)
        ax.set_title("Detection Count Over Time", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved: {output_path}")

    def generate_all_plots(self, metrics_files: List[str]):
        """Generate all comparison plots."""
        print(f"\n{'='*60}")
        print(f"📊 Generating Visualizations")
        print(f"{'='*60}")
        print(f"Input files: {len(metrics_files)}")
        print(f"Output directory: {self.output_dir}")
        if self.skip_warmup > 0:
            print(f"Skipping warmup: {self.skip_warmup} frames")
        if self.remove_outliers:
            print(f"Outlier removal: Enabled (IQR method)")
        print()

        # Load all metrics
        metrics_list = []
        for filepath in metrics_files:
            try:
                metrics = self.load_metrics(filepath)
                metrics_list.append(metrics)
                print(f"✅ Loaded: {Path(filepath).name} ({metrics['name']})")
            except Exception as e:
                print(f"⚠️  Failed to load {filepath}: {e}")

        if not metrics_list:
            print("\n❌ No valid metrics files loaded")
            return

        print(f"\n{'='*60}")
        print("Generating plots...")
        print(f"{'='*60}\n")

        # Generate all plots
        try:
            self.plot_summary_comparison(metrics_list)
            self.plot_latency_over_time(metrics_list)
            self.plot_fps_over_time(metrics_list)
            self.plot_latency_distribution(metrics_list)
            self.plot_percentile_comparison(metrics_list)
            self.plot_detection_over_time(metrics_list)

            print(f"\n{'='*60}")
            print(f"✅ All plots generated successfully!")
            print(f"{'='*60}")
            print(f"\n📁 Output directory: {self.output_dir.absolute()}")

        except Exception as e:
            print(f"\n❌ Error generating plots: {e}")
            import traceback

            traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced metrics visualization with warmup handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with warmup skip (recommended)
  python visualize_metrics.py --glob "metrics/*.json" --skip-warmup 15

  # Without outlier removal
  python visualize_metrics.py --glob "metrics/*.json" --no-remove-outliers

  # Windows PowerShell
  python visualize_metrics.py --glob "metrics/*.json" --skip-warmup 10 --output plots/
        """,
    )

    parser.add_argument("metrics_files", nargs="*", type=str, help="Paths to metrics JSON files")
    parser.add_argument("--glob", "-g", type=str, default=None, help="Glob pattern to match metrics files")
    parser.add_argument(
        "--output", "-o", type=str, default="plots", help="Output directory for plots (default: plots/)"
    )
    parser.add_argument(
        "--skip-warmup", "-s", type=int, default=0, help="Skip first N frames (warmup period). Recommended: 10-15"
    )
    parser.add_argument(
        "--no-remove-outliers", action="store_true", help="Disable automatic outlier removal in distributions"
    )

    args = parser.parse_args()

    # Get files
    if args.glob:
        valid_files = glob_module.glob(args.glob)
        if not valid_files:
            print(f"❌ No files found matching pattern: {args.glob}")
            sys.exit(1)
    else:
        valid_files = [f for f in args.metrics_files if Path(f).exists()]

    if not valid_files:
        print("\n❌ No valid metrics files provided")
        print('\nTip: python visualize_metrics.py --glob "metrics/*.json" --skip-warmup 15')
        sys.exit(1)

    # Create visualizer
    visualizer = MetricsVisualizer(
        output_dir=args.output, skip_warmup=args.skip_warmup, remove_outliers=not args.no_remove_outliers
    )
    visualizer.generate_all_plots(valid_files)


if __name__ == "__main__":
    main()
