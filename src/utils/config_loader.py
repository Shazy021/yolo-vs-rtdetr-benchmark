from pathlib import Path
from typing import Any, Optional

import yaml


class Config:
    """
    Configuration manager with support for YAML files and CLI overrides.

    Attributes:
        data: Dictionary containing all configuration values
        config_path: Path to the loaded config file
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n" f"Create it using: cp config.example.yaml config.yaml"
            )

        try:
            with open(self.config_path, "r") as f:
                self.data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML config: {e}")

        # Validate required sections
        self._validate()

    def _validate(self):
        """Validate config structure."""
        required_sections = ["models", "inference", "video", "export", "metrics"]

        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required config section: {section}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get config value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., 'inference.conf_threshold')
            default: Default value if key not found

        Returns:
            Config value or default

        Example:
            >>> config = Config()
            >>> conf = config.get('inference.conf_threshold', 0.25)
        """
        keys = key_path.split(".")
        value = self.data

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_model_path(self, model: str, backend: str) -> str:
        """
        Get model weights path.

        Args:
            model: Model type ('yolo' or 'rtdetr')
            backend: Backend type ('pytorch', 'onnx', 'tensorrt')

        Returns:
            Path to model weights

        Raises:
            KeyError: If model/backend combination not found
        """
        try:
            return self.data["models"][model][backend]
        except KeyError:
            raise KeyError(f"Model path not found: models.{model}.{backend}")

    def get_input_size(self, video_path: Optional[str] = None) -> tuple:
        """
        Get input size based on configuration mode.

        Args:
            video_path: Path to video file (used for adaptive mode)

        Returns:
            Tuple of (height, width)
        """
        mode = self.get("inference.input_size.mode", "fixed")

        if mode == "fixed":
            size = self.get("inference.input_size.fixed_size", [640, 640])
            return tuple(size)
        elif mode == "adaptive" and video_path:
            # Import here to avoid circular dependency
            from utils import get_video_optimal_size

            try:
                info = get_video_optimal_size(video_path)
                return (info["optimal_height"], info["optimal_width"])
            except Exception:
                return (640, 640)
        else:
            return (640, 640)

    def merge_cli_args(self, args) -> None:
        """
        Merge CLI arguments into config (CLI takes priority).

        Args:
            args: argparse.Namespace object with CLI arguments
        """
        # Map CLI args to config paths
        cli_mappings = {
            "conf": "inference.conf_threshold",
            "nms": "inference.nms_threshold",
            "max_frames": "video.max_frames",
            "show": "video.show_preview",
            "no_display_info": "video.display_info",  # Inverted
        }

        for cli_key, config_path in cli_mappings.items():
            if hasattr(args, cli_key):
                cli_value = getattr(args, cli_key)

                # Skip None values (not provided)
                if cli_value is None:
                    continue

                # Handle inverted flags
                if cli_key == "no_display_info":
                    cli_value = not cli_value

                # Set value in config
                self._set_nested(config_path, cli_value)

    def _set_nested(self, key_path: str, value: Any):
        """Set nested dictionary value using dot notation."""
        keys = key_path.split(".")
        d = self.data

        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]

        d[keys[-1]] = value

    def save(self, output_path: str):
        """
        Save current config to YAML file.

        Args:
            output_path: Path to output YAML file
        """
        with open(output_path, "w") as f:
            yaml.dump(self.data, f, default_flow_style=False, sort_keys=False)

        print(f"💾 Config saved to: {output_path}")

    def print_summary(self):
        """Print configuration summary."""
        print("\n" + "=" * 60)
        print("⚙️  Configuration Summary")
        print("=" * 60)
        print(f"Config file: {self.config_path}")
        print(f"\nInference:")
        print(f"  Confidence threshold: {self.get('inference.conf_threshold')}")
        print(f"  NMS threshold: {self.get('inference.nms_threshold')}")
        print(f"  Input size mode: {self.get('inference.input_size.mode')}")
        print(f"  GPU enabled: {self.get('inference.device.use_gpu')}")
        print(f"\nVideo:")
        print(f"  Show preview: {self.get('video.show_preview')}")
        print(f"  Display info: {self.get('video.display_info')}")
        print(f"  Max frames: {self.get('video.max_frames') or 'all'}")
        print(f"\nMetrics:")
        print(f"  Enabled: {self.get('metrics.enabled')}")
        print(f"  Save to file: {self.get('metrics.save_to_file')}")
        print("=" * 60 + "\n")

    def __repr__(self) -> str:
        """String representation."""
        return f"Config(path='{self.config_path}', sections={list(self.data.keys())})"


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Config object
    """
    return Config(config_path)
