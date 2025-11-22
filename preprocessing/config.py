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
    """Configuration for data augmentation during preprocessing."""

    enabled: bool = False
    rotation_degrees: float = 10.0
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.1
    probability: float = 0.5


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


@dataclass
class FaceDetectionConfig:
    """Configuration for YuNet face detector."""

    model_path: Path = models_dir / "face_detection_yunet_2023mar.onnx"
    input_size: tuple[int, int] = (768, 576)
    score_threshold: float = 0.9
    nms_threshold: float = 0.3
    top_k: int = 5000

    # Assert model file exists at config creation
    assert model_path.exists(), (
        f"Model file not found: {model_path}. "
        "Please run scripts/download-weights.py to download the model weights."
    )
