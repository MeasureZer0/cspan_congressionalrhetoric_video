# Video Preprocessing Pipeline

This module handles video preprocessing for the congressional rhetoric project, including face detection, cropping, resizing, and optical flow computation.

## Quick Start

Run preprocessing with default settings:

```bash
python preprocessing/preprocess.py
```

Run with augmentation enabled:

```bash
python preprocessing/preprocess.py --use_augmentation
```

Run with custom settings:

```bash
python preprocessing/preprocess.py \
  --frame_skip 15 \
  --use_augmentation \
  --augmentation_probability 0.7 \
  --rotation_degrees 15.0
```

## Configuration System

The preprocessing pipeline uses a clean, typed configuration system with three main config classes:

### `PreprocessingConfig`

Main configuration for the preprocessing pipeline:

- **Paths**: `data_dir`, `label_file`, `out_dir`
- **Processing**: `frame_skip`, `size`, `margin`, `crop_width_ratio`
- **Execution**: `purge`, `max_workers`
- **Augmentation**: `augmentation` (AugmentationConfig)

### `AugmentationConfig`

Configuration for data augmentation:

- `enabled`: Whether to use augmentation (default: False)
- `rotation_degrees`: Max rotation in degrees (default: 10.0)
- `brightness`: Brightness jitter factor (default: 0.2)
- `contrast`: Contrast jitter factor (default: 0.2)
- `saturation`: Saturation jitter factor (default: 0.2)
- `hue`: Hue jitter factor (default: 0.1)
- `probability`: Probability of augmenting each video (default: 0.5)

### `FaceDetectionConfig`

Configuration for YuNet face detector:

- `model_path`: Path to ONNX model
- `input_size`: Input size for detector (default: 768x576)
- `score_threshold`: Confidence threshold (default: 0.9)
- `nms_threshold`: NMS threshold (default: 0.3)
- `top_k`: Max detections (default: 5000)

## Command Line Arguments

### Input/Output

- `--data_dir PATH`: Directory containing raw videos
- `--label_file PATH`: CSV file with video labels
- `--out_dir PATH`: Output directory for preprocessed tensors

### Processing Parameters

- `--frame_skip N`: Process every N-th frame (default: 30)
- `--size H W`: Target size for face tensors (default: 224 224)
- `--margin FLOAT`: Margin around detected faces (default: 0.1)
- `--crop_width_ratio FLOAT`: Width ratio for center crop (default: 0.5)

### Execution Options

- `--purge`: Delete existing outputs before processing
- `--max_workers N`: Max parallel workers (default: CPU count)

### Augmentation Options

- `--use_augmentation`: Enable augmentation
- `--rotation_degrees FLOAT`: Max rotation (default: 10.0)
- `--brightness FLOAT`: Brightness jitter (default: 0.2)
- `--contrast FLOAT`: Contrast jitter (default: 0.2)
- `--saturation FLOAT`: Saturation jitter (default: 0.2)
- `--hue FLOAT`: Hue jitter (default: 0.1)
- `--augmentation_probability FLOAT`: Aug probability (default: 0.5)

## Pipeline Overview

1. **Extract frames** from video at specified `frame_skip` rate
2. **Center crop** frames by `crop_width_ratio` to focus on speaker
3. **Detect faces** using YuNet detector
4. **Crop and resize** detected faces with optional margin
5. **Apply augmentation** (if enabled) with consistent params per video
6. **Compute optical flow** between augmented consecutive frames
7. **Save tensors** as `.pt` files (faces and flows)

## Augmentation Behavior

When augmentation is enabled:

- Each video has a `probability` chance of being augmented
- If augmented, **all frames in the video get the same transformation**
  - Same rotation angle
  - Same brightness/contrast/saturation/hue adjustments
- Optical flow is computed **after** augmentation
- This ensures temporal consistency within each video

## Output Structure

```
data/faces/frame_skip_30/
├── 578982906_faces.pt
├── 578982906_flows.pt
├── 578984029_faces.pt
├── 578984029_flows.pt
└── ...
```

Each file contains:

- `*_faces.pt`: Tensor of shape `(N, 3, H, W)` with N face frames
- `*_flows.pt`: Tensor of shape `(N, 2, H, W)` with N-1 optical flows

## Module Structure

```
preprocessing/
├── config.py              # Configuration dataclasses
├── preprocess.py          # Main CLI entry point
├── crop_faces.py          # Core preprocessing logic
├── frame_augmentation.py  # Augmentation implementation
├── extract_frames.py      # Video frame extraction
├── raft_optical_flow.py   # Optical flow computation
└── README.md              # This file
```

## Examples

### Basic preprocessing without augmentation

```bash
python preprocessing/preprocess.py
```

### Preprocessing with augmentation

```bash
python preprocessing/preprocess.py \
  --use_augmentation \
  --augmentation_probability 0.5
```

### Aggressive augmentation

```bash
python preprocessing/preprocess.py \
  --use_augmentation \
  --augmentation_probability 0.8 \
  --rotation_degrees 15.0 \
  --brightness 0.3 \
  --contrast 0.3
```

## Notes

- Augmentation is applied **before** optical flow computation for realistic flow data
- All frames in a video receive identical augmentation parameters for temporal consistency
- The face detector focuses on the most central face in each frame
- Videos without detectable faces are skipped with a warning
- Processing is parallelized across videos using `ProcessPoolExecutor`
