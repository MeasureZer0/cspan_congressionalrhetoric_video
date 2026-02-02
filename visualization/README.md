# Visualization Tools

This directory contains scripts for visualizing data used in the project.

## Scripts

### `peek_faces.py`

This script allows you to visualize the face tensors stored in `.pt` files generated during preprocessing. It creates a grid of images from the frames in the tensor.

**Usage:**

```bash
python visualization/peek_faces.py --input-path <path_to_pt_file> [options]
```

**Arguments:**

* `--input-path`: Path to a `.pt` file containing face tensors, or a directory. If a directory is provided, it lists the available `.pt` files.
* `--output`, `-o`: Path where the output image will be saved. Default is `preview.png`.
* `--nrow`: Number of images per row in the output grid. Default is 8.
* `--max-frames`: Maximum number of frames to include in the visualization. Default is 64.
* `--random`: If set, randomly selects frames from the tensor instead of taking the first `max-frames`.

**Examples:**

To visualize the first 64 frames from a specific file and save it to `faces_preview.png`:

```bash
python visualization/peek_faces.py --input-path data/faces/578982906_faces.pt --output faces_preview.png
```

To visualize a random selection of 16 frames in a 4x4 grid:

```bash
python visualization/peek_faces.py --input-path data/faces/578982906_faces.pt --max-frames 16 --nrow 4 --random
```
