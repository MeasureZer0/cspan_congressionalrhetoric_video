# Scripts

Utility scripts for setup and local data management.

## `download-weights.py`

Downloads the required model weights (currently the YuNet face detector) into
`data/weights/` relative to the project root.

Usage:

```bash
python scripts/download-weights.py
```

Notes:

- Creates `data/weights/` automatically if it does not exist.
- Shows a progress bar while downloading.

## `pip-uninstall.py`

Uninstalls one or more packages and recursively collects their dependencies,
removing only packages that are not required by anything else currently
installed.

Usage:

```bash
python scripts/pip-uninstall.py pkg1 [pkg2 ...]
```

Notes:

- Runs `pip uninstall -y ...` automatically.
- Regenerates `requirements.txt` in the current working directory using
  `pip freeze` after uninstalling.
- This is most useful inside your project virtual environment.

## `move-random-videos.py`

Moves or copies a random subset of video files from a source directory to a
destination directory.

Supported extensions: `.mp4`, `.mkv`, `.avi`, `.mov`, `.flv`, `.wmv`, `.webm`

Usage:

```bash
python scripts/move-random-videos.py SOURCE_DIR DEST_DIR [options]
```

Common options:

- `-n, --count <int>`: number of files to select (default: `1000`)
- `-r, --recursive`: search subdirectories recursively
- `--copy`: copy files instead of moving them
- `--overwrite`: overwrite files that already exist in the destination
- `--dry-run`: print planned actions without modifying files

Notes:

- Both source and destination directories must already exist.
- If fewer videos exist than requested, it processes all available videos.

## `label-videos.py`

Interactive Tk + VLC video labeling tool. It plays videos from a folder and lets you label each one as `positive`, `neutral`, or `negative`.

Usage:

```bash
python scripts/label-videos.py FOLDER [options]
```

Common options:

- `--csv <path>`: output CSV path (default: `labels.csv`)
- `--width <int>`: display width in pixels (default: `960`)
- `--height <int>`: display height in pixels (default: `540`)

Notes:

- Videos are played in alphabetical order.
- The CSV file is updated immediately as you label videos.
- Output CSV format is `"filename","label"` (quoted 2-column CSV).
- `Back` returns to the previous video index so you can relabel it.
- Keyboard shortcuts: `p` positive, `u` neutral, `n` negative, `s` skip, `b` back, `q` quit, `Space` pause/resume.
- Requires `python-vlc` and a local VLC installation (all platforms).
- Linux/Windows: video is embedded in the app window via VLC.
- macOS: due to Tk + VLC `NSView` embedding behavior, video plays in a separate VLC/Tk window and the labeling controls stay in a floating controls window.
- Requires Tkinter (`tk`). On macOS with Homebrew Python, install Tk support if needed.
- If VLC cannot attach to Tk on macOS, you may need to point `TKVLC_LIBTK_PATH` at your `libtk*.dylib`.
