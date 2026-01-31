"""Configuration management for video preprocessing pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Define paths to input data relative to this file
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
data_dir = project_root / "data"
raw_videos_dir = data_dir / "raw_videos"
label_file = data_dir / "labels.csv"
models_dir = data_dir / "weights"


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation during preprocessing.

    When enabled, creates augmented versions of videos as additional training samples.
    Original videos are always kept - augmented versions are saved with '_aug' suffix.
    """

    enabled: bool = False
    rotation_degrees: float = 10.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1


@dataclass
class PreprocessingConfig:
    """Configuration for video preprocessing pipeline."""

    # Input/output paths
    data_dir: Path = raw_videos_dir
    label_file: Path = label_file

    # Processing parameters
    frame_skip: int = 30
    out_dir: Path = data_dir / "faces" / f"frame_skip_{frame_skip}"
    size: tuple[int, int] = (224, 224)
    margin: float = 0.1
    crop_width_ratio: float = 0.5

    # Execution options
    purge: bool = False
    max_workers: Optional[int] = None

    # Augmentation
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    @classmethod
    def load(cls, **overrides: dict[str, object]) -> "PreprocessingConfig":
        """Load configuration from defaults and apply non-None overrides."""
        config = cls()

        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)

        return config

    def __post_init__(self) -> None:
        assert self.label_file.exists(), f"Label file not found: {self.label_file}"
        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"

    def __str__(self) -> str:
        """String representation of the configuration."""
        lines = [
            f"Data dir: {self.data_dir}",
            f"Label file: {self.label_file}",
            f"Output dir: {self.out_dir}",
            f"Frame skip: {self.frame_skip}",
            f"Size: {self.size}",
            f"Margin: {self.margin}",
            f"Crop width ratio: {self.crop_width_ratio}",
            f"Purge: {self.purge}",
            f"Max workers: {self.max_workers}",
            f"Augmentation enabled: {self.augmentation.enabled}",
        ]
        if self.augmentation.enabled:
            lines.extend(
                [
                    f"  Rotation degrees: ±{self.augmentation.rotation_degrees}",
                    f"  Brightness: ±{self.augmentation.brightness}",
                    f"  Contrast: ±{self.augmentation.contrast}",
                    f"  Saturation: ±{self.augmentation.saturation}",
                    f"  Hue: ±{self.augmentation.hue}",
                ]
            )
        return "\n".join(lines)


@dataclass
class FaceDetectionConfig:
    """Configuration for YuNet face detector."""

    model_path: Path = models_dir / "face_detection_yunet_2023mar.onnx"
    input_size: tuple[int, int] = (768, 576)
    score_threshold: float = 0.9
    nms_threshold: float = 0.3
    top_k: int = 5000

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Please run scripts/download-weights.py to download the model weights."
            )
