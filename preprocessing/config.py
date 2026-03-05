"""Configuration management for video preprocessing pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
LABEL_FILE = DATA_DIR / "labels.csv"
MODELS_DIR = DATA_DIR / "weights"


@dataclass
class PreprocessingConfig:
    """Configuration for video preprocessing pipeline."""

    # Input/output paths
    data_dir: Path = RAW_VIDEOS_DIR
    label_file: Path = LABEL_FILE

    # Processing parameters
    frame_skip: int = 30
    out_dir: Optional[Path] = None
    size: tuple[int, int] = (128, 128)
    margin: float = 0.1
    crop_width_ratio: float = 0.5

    # Execution options
    purge: bool = False
    max_workers: Optional[int] = None

    @classmethod
    def load(cls, **overrides: object) -> "PreprocessingConfig":
        """Load configuration from defaults and apply non-None overrides."""
        config = cls()
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        config.__post_init__()
        return config

    def __post_init__(self) -> None:
        if self.out_dir is None:
            self.out_dir = DATA_DIR / "processed" / f"frame_skip_{self.frame_skip}"
        assert self.label_file.exists(), f"Label file not found: {self.label_file}"
        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Data dir: {self.data_dir}",
                f"Label file: {self.label_file}",
                f"Output dir: {self.out_dir}",
                f"Frame skip: {self.frame_skip}",
                f"Size: {self.size}",
                f"Margin: {self.margin}",
                f"Crop width ratio: {self.crop_width_ratio}",
                f"Purge: {self.purge}",
                f"Max workers: {self.max_workers}",
            ]
        )


@dataclass
class FaceDetectionConfig:
    """Configuration for YuNet face detector."""

    model_path: Path = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    input_size: tuple[int, int] = (768, 576)
    score_threshold: float = 0.9
    nms_threshold: float = 0.3
    top_k: int = 5000

    def __post_init__(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Please run scripts/download-weights.py to download the model weights."
            )
