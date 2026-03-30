#!/usr/bin/env python3
"""Tk + VLC video labeler.

FIXES:
  - Case-insensitive filename matching (video.mp4 == VIDEO.MP4)
  - No duplicate labels in CSV (rewrites entire file on each label)
  - Preserves original filename capitalization in CSV
  - Saves timestamp to CSV
  - Dynamically skips already-labeled videos during navigation (_advance + _load_current)
  - Skips labeled videos even when encountered via _back → re-advance

# Based heavily on https://github.com/oaubert/python-vlc/blob/master/examples/tkvlc.py

Architecture:
  - root.mainloop() owns the UI thread — never blocked.
  - All VLC calls (stop / set_media / play) run in a single daemon worker thread.
  - Communication back to Tk uses root.after(0, callback) which is thread-safe.
"""

import argparse
import csv
import os
import queue
import sys
import threading
import time
import tkinter as tk
from ctypes import c_void_p, cdll
from ctypes.util import find_library
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

if sys.platform.startswith("win"):
    VLC_PATH = r"C:\Program Files\VideoLAN\VLC"
    os.add_dll_directory(VLC_PATH)
    os.environ["PYTHON_VLC_MODULE_PATH"] = VLC_PATH

import vlc

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
IS_MAC = sys.platform.startswith("darwin")
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")
SEEK_STEP_MS = 5_000
TICK_INTERVAL_MS = 20


def format_ms(ms: int) -> str:
    total_seconds = max(0, ms // 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _build_nsview_getter() -> Any:  # noqa: ANN401
    if not IS_MAC:
        return lambda _widget_id: None
    tk_ver = tk.TkVersion
    lib_name = f"libtk{tk_ver}.dylib"
    candidates: List[str] = []
    for prefix in (getattr(sys, "base_prefix", ""), sys.prefix):
        if prefix:
            candidates.append(os.path.join(prefix, "lib", lib_name))
    found = find_library(f"tk{tk_ver}") or find_library("tk")
    if found:
        candidates.append(found)
    env = os.environ.get("TKVLC_LIBTK_PATH", "")
    for item in env.split(os.pathsep):
        if not item:
            continue
        item = os.path.expanduser(item)
        candidates.append(item)
        if not item.endswith(lib_name):
            candidates.append(os.path.join(item, lib_name))
    for path in candidates:
        try:
            if not path:
                continue
            lib = cdll.LoadLibrary(path)
            getter = lib.TkMacOSXGetRootControl
            getter.restype = c_void_p
            getter.argtypes = (c_void_p,)
            return getter
        except Exception:
            continue
    return lambda _widget_id: None


_GET_NSVIEW = _build_nsview_getter()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def list_videos(folder: Path) -> List[Path]:
    videos = [
        p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    videos.sort(key=lambda p: p.name.lower())
    return videos


def read_labels(csv_path: Path) -> Dict[str, Tuple[str, str, str]]:
    """
    Read labels from CSV with case-insensitive lookup.

    Handles:
      - UTF-8 BOM (utf-8-sig)
      - Missing timestamp column
      - Headerless CSVs (treats first two columns as filename, label)

    Returns:
        Dict mapping lowercase_filename -> (original_filename, label, timestamp)
    """
    labels: Dict[str, Tuple[str, str, str]] = {}
    if not csv_path.exists():
        return labels

    # utf-8-sig strips BOM if present, harmless otherwise
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        raw = f.read()

    if not raw.strip():
        return labels

    lines = raw.splitlines()
    first = lines[0].lower()

    # Detect whether the first row is a header or real data.
    # A header contains the word "filename" in some column.
    has_header = "filename" in first

    reader = csv.reader(lines)
    if has_header:
        next(reader)  # skip header row

    for row in reader:
        if len(row) < 2:
            continue
        filename = row[0].strip()
        label = row[1].strip()

        if filename and label:
            labels[filename.lower()] = (filename, label)

    return labels


def write_labels(csv_path: Path, labels: Dict[str, Tuple[str, str, str]]) -> None:
    """
    Write all labels to CSV (filename + label + timestamp), replacing the entire file.
    Prevents duplicates when re-labeling the same video.

    Args:
        csv_path: Path to CSV file
        labels: Dict mapping lowercase_filename -> (original_filename, label, timestamp)
    """
    tmp_path = csv_path.with_name(csv_path.name + ".tmp")

    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "label"],
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()

        for filename_lower in sorted(labels.keys()):
            original_filename, label = labels[filename_lower]
            writer.writerow(
                {
                    "filename": original_filename,
                    "label": label,
                }
            )

    # Atomic replace — on Windows, target must not exist first
    if IS_WIN and csv_path.exists():
        csv_path.unlink()
    tmp_path.replace(csv_path)


# ---------------------------------------------------------------------------
# VLC worker thread
# ---------------------------------------------------------------------------


class VLCWorker(threading.Thread):
    """Serialises all blocking VLC calls onto a single background thread.

    Commands are plain dicts:
        {"cmd": "load",  "path": Path, "attach_fn": callable, "done": callable, "error_fn": callable}
        {"cmd": "stop",  "done": callable | None}
        {"cmd": "seek",  "ms": int}
        {"cmd": "pause"}
        {"cmd": "quit"}

    Callbacks (done / error_fn / attach_fn) are always dispatched back to the
    Tk event loop via root.after(0, ...) so they are safe to touch widgets.
    """

    def __init__(self, root: tk.Tk) -> None:
        super().__init__(daemon=True, name="vlc-worker")
        self.root = root
        self._q: queue.Queue = queue.Queue()
        self.instance = vlc.Instance()
        self.player: Any = self.instance.media_player_new()

    def submit(self, cmd: dict) -> None:
        self._q.put(cmd)

    def run(self) -> None:
        while True:
            try:
                cmd = self._q.get(timeout=1)
            except queue.Empty:
                continue

            action = cmd.get("cmd")

            if action == "quit":
                try:
                    self.player.stop()
                    self.player.release()
                    self.instance.release()
                except Exception:
                    pass
                break

            elif action == "stop":
                try:
                    self.player.stop()
                except Exception:
                    pass
                done = cmd.get("done")
                if done:
                    self.root.after(0, done)

            elif action == "load":
                path: Path = cmd["path"]
                attach_fn = cmd.get("attach_fn")
                done = cmd.get("done")
                error_fn = cmd.get("error_fn")
                try:
                    self.player.stop()
                    media = self.instance.media_new(str(path))
                    self.player.set_media(media)
                    if attach_fn:
                        self.root.after(0, attach_fn)
                    time.sleep(0.05)
                    result = self.player.play()
                    if result == -1:
                        raise RuntimeError(f"VLC refused to open: {path}")
                    if done:
                        self.root.after(0, done)
                except Exception as exc:
                    if error_fn:
                        self.root.after(0, lambda e=exc: error_fn(e))

            elif action == "seek":
                try:
                    target_ms: int = cmd["ms"]
                    length_ms = self.player.get_length()
                    if length_ms and length_ms > 0:
                        target_ms = min(max(0, target_ms), length_ms - 250)
                    self.player.set_time(max(0, target_ms))
                except Exception:
                    pass

            elif action == "pause":
                try:
                    self.player.pause()
                except Exception:
                    pass

            self._q.task_done()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------


class VideoLabelerApp:
    def __init__(
        self,
        folder: Path,
        csv_path: Path,
        width: int = 960,
        height: int = 540,
        purge: bool = False,
    ) -> None:
        self.folder = folder
        self.csv_path = csv_path
        self.labels = read_labels(csv_path)
        self.all_videos = list_videos(folder)
        if not self.all_videos:
            raise RuntimeError(f"No videos found in {folder}")

        if purge:
            removed_any = False
            for video in self.all_videos:
                if video.name.lower() in self.labels:
                    del self.labels[video.name.lower()]
                    removed_any = True
            if removed_any:
                write_labels(self.csv_path, self.labels)

        # Filter out already-labeled videos at startup using the CSV.
        # Case-insensitive: video.mp4 == VIDEO.MP4
        self.videos = [v for v in self.all_videos if v.name.lower() not in self.labels]

        if not self.videos:
            raise RuntimeError(
                f"No unlabeled videos found in {folder}. Use --purge to relabel."
            )

        self.index = 0

        self.history: List[int] = []
        self.current_path: Optional[Path] = None
        self.paused = False
        self._slider_dragging = False
        self._loading = False

        # ------------------------------------------------------------------
        # Build UI
        # ------------------------------------------------------------------
        self.root = tk.Tk()
        self.root.title("Video Labeler (TkVLC)")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.slider_var = tk.DoubleVar(master=self.root, value=0.0)
        self.timeline_var = tk.StringVar(master=self.root, value="0:00 / 0:00")
        self.status_var = tk.StringVar(master=self.root, value="")

        self.video_window: Optional[tk.Toplevel] = None
        video_parent: tk.Misc = self.root
        if IS_MAC:
            self.root.title("Label Controls")
            self.video_window = tk.Toplevel(self.root)
            self.video_window.title("Video Labeler (Video)")
            self.video_window.geometry(f"{width + 20}x{height + 20}")
            self.video_window.columnconfigure(0, weight=1)
            self.video_window.rowconfigure(0, weight=1)
            self.video_window.protocol("WM_DELETE_WINDOW", self._quit)
            video_parent = self.video_window
        else:
            self.root.geometry(f"{max(700, width + 40)}x{height + 220}")

        self.video_frame = ttk.Frame(video_parent)
        self.video_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.video_frame.rowconfigure(0, weight=1)
        self.video_frame.columnconfigure(0, weight=1)

        self.video_canvas = tk.Canvas(
            self.video_frame, width=width, height=height, highlightthickness=0
        )
        self.video_canvas.grid(row=0, column=0, sticky="nsew")

        self.info_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.info_var).grid(
            row=1, column=0, padx=10, pady=(0, 2), sticky="w"
        )

        self.progress_var = tk.StringVar()
        ttk.Label(self.root, textvariable=self.progress_var, foreground="#007acc").grid(
            row=2, column=0, padx=10, pady=(0, 2), sticky="w"
        )

        ttk.Label(self.root, textvariable=self.status_var, foreground="gray").grid(
            row=3, column=0, padx=10, pady=(0, 4), sticky="w"
        )

        timeline = ttk.Frame(self.root)
        timeline.grid(row=4, column=0, padx=10, pady=(0, 8), sticky="ew")
        timeline.columnconfigure(0, weight=1)

        self.timeline_scale = tk.Scale(
            timeline,
            from_=0,
            to=1000,
            orient="horizontal",
            showvalue=False,
            highlightthickness=0,
            variable=self.slider_var,
            command=self.on_slider_change,
        )
        self.timeline_scale.grid(row=0, column=0, sticky="ew")
        self.timeline_scale.bind("<ButtonPress-1>", self.on_slider_press)
        self.timeline_scale.bind("<B1-Motion>", self.on_slider_drag)
        self.timeline_scale.bind("<ButtonRelease-1>", self.on_slider_release)

        ttk.Label(timeline, textvariable=self.timeline_var).grid(
            row=1, column=0, sticky="e"
        )

        playback_row = ttk.Frame(self.root)
        playback_row.grid(row=5, column=0, padx=10, pady=(0, 8), sticky="ew")
        for col in range(3):
            playback_row.columnconfigure(col, weight=1)

        ttk.Button(
            playback_row,
            text="Back 5s [Left]",
            command=lambda: self.seek_relative(-SEEK_STEP_MS),
        ).grid(row=0, column=0, padx=4, sticky="ew")
        ttk.Button(playback_row, text="Pause [Space]", command=self.toggle_pause).grid(
            row=0, column=1, padx=4, sticky="ew"
        )
        ttk.Button(
            playback_row,
            text="Forward 5s [Right]",
            command=lambda: self.seek_relative(SEEK_STEP_MS),
        ).grid(row=0, column=2, padx=4, sticky="ew")

        label_row = ttk.Frame(self.root)
        label_row.grid(row=6, column=0, padx=10, pady=(0, 10), sticky="ew")
        for col in range(6):
            label_row.columnconfigure(col, weight=1)

        self._label_buttons: List[ttk.Button] = []

        def _btn(text: str, cmd, col: int) -> ttk.Button:
            b = ttk.Button(label_row, text=text, command=cmd)
            b.grid(row=0, column=col, padx=4, sticky="ew")
            self._label_buttons.append(b)
            return b

        _btn("Positive [P]", lambda: self._label_and_advance("positive"), 0)
        _btn("Neutral [U]", lambda: self._label_and_advance("neutral"), 1)
        _btn("Negative [N]", lambda: self._label_and_advance("negative"), 2)
        _btn("Skip [S]", self._skip, 3)
        _btn("Back [B]", self._back, 4)
        _btn("Quit [Q]", self._quit, 5)

        self._bind_keys(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        if self.video_window is not None:
            self._bind_keys(self.video_window)

        self.root.update_idletasks()
        if self.video_window is not None:
            self.video_window.update_idletasks()
            try:
                self.root.transient(self.video_window)
            except tk.TclError:
                pass
            try:
                self.root.attributes("-topmost", True)
            except tk.TclError:
                pass
            self.root.resizable(False, False)
            vx = self.video_window.winfo_x()
            vy = self.video_window.winfo_y()
            vh = self.video_window.winfo_height()
            vw = self.video_window.winfo_width()
            rw = self.root.winfo_width()
            rh = self.root.winfo_height()
            self.root.geometry(f"+{vx + vw // 2 - rw // 2}+{vy + vh + rh // 2}")

        # ------------------------------------------------------------------
        # Start VLC worker
        # ------------------------------------------------------------------
        self._vlc = VLCWorker(self.root)
        self._vlc.start()

    # ------------------------------------------------------------------
    # Label-aware index helpers
    # ------------------------------------------------------------------

    def _is_labeled(self, idx: int) -> bool:
        """Return True if the video at index idx is already labeled."""
        if idx < 0 or idx >= len(self.videos):
            return False
        return self.videos[idx].name.lower() in self.labels

    def _first_unlabeled_from(self, start: int) -> int:
        """Return the first index >= start whose video is not labeled."""
        idx = start
        while idx < len(self.videos) and self._is_labeled(idx):
            idx += 1
        return idx

    def _count_labeled(self) -> int:
        return sum(1 for v in self.videos if v.name.lower() in self.labels)

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _bind_keys(self, widget: tk.Misc) -> None:
        widget.bind("<KeyPress-p>", lambda _e: self._label_and_advance("positive"))
        widget.bind("<KeyPress-u>", lambda _e: self._label_and_advance("neutral"))
        widget.bind("<KeyPress-n>", lambda _e: self._label_and_advance("negative"))
        widget.bind("<KeyPress-s>", lambda _e: self._skip())
        widget.bind("<KeyPress-b>", lambda _e: self._back())
        widget.bind("<KeyPress-q>", lambda _e: self._quit())
        widget.bind("<Left>", lambda _e: self.seek_relative(-SEEK_STEP_MS))
        widget.bind("<Right>", lambda _e: self.seek_relative(SEEK_STEP_MS))
        widget.bind("<space>", lambda _e: self.toggle_pause())

    # ------------------------------------------------------------------
    # Button enable / disable
    # ------------------------------------------------------------------

    def _set_buttons_state(self, state: str) -> None:
        for btn in self._label_buttons:
            try:
                btn.configure(state=state)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Player window attachment  (must run on main thread)
    # ------------------------------------------------------------------

    def _attach_player_to_widget(self) -> None:
        self.root.update_idletasks()
        widget_id = self.video_canvas.winfo_id()
        player = self._vlc.player
        if IS_WIN:
            player.set_hwnd(widget_id)
        elif IS_MAC:
            nsview = _GET_NSVIEW(widget_id)
            if nsview:
                player.set_nsobject(nsview)
        else:
            player.set_xwindow(widget_id)

    # ------------------------------------------------------------------
    # Playback helpers (UI thread)
    # ------------------------------------------------------------------

    def toggle_pause(self) -> None:
        if self.current_path is None or self._loading:
            return
        self.paused = not self.paused
        self._vlc.submit({"cmd": "pause"})

    def seek_to_ms(self, target_ms: int) -> None:
        if self.current_path is None or self._loading:
            return
        self._vlc.submit({"cmd": "seek", "ms": target_ms})
        self.slider_var.set(target_ms)
        self.update_timeline_label(target_ms)

    def seek_relative(self, delta_ms: int) -> None:
        if self.current_path is None or self._loading:
            return
        current_ms = self._vlc.player.get_time()
        if current_ms is None or current_ms < 0:
            current_ms = 0
        self.seek_to_ms(max(0, current_ms + delta_ms))

    # ------------------------------------------------------------------
    # Timeline UI
    # ------------------------------------------------------------------

    def update_timeline_label(
        self, current_ms: int, length_ms: Optional[int] = None
    ) -> None:
        if length_ms is None:
            length_ms = self._vlc.player.get_length()
        length_text = format_ms(length_ms) if length_ms and length_ms > 0 else "0:00"
        self.timeline_var.set(f"{format_ms(current_ms)} / {length_text}")

    def sync_timeline(self) -> None:
        if self.current_path is None or self._slider_dragging or self._loading:
            return
        current_ms = self._vlc.player.get_time()
        if current_ms is None or current_ms < 0:
            current_ms = 0
        length_ms = self._vlc.player.get_length()
        if length_ms is not None and length_ms > 0:
            self.timeline_scale.configure(to=length_ms)
            current_ms = min(current_ms, length_ms)
        self.slider_var.set(current_ms)
        self.update_timeline_label(current_ms, length_ms)

    def update_info(self) -> None:
        if self.current_path is None:
            return
        basename = self.current_path.name
        existing = self.labels.get(basename.lower())
        existing_text = f" | ✔ już oznaczony: {existing[1]}" if existing else ""
        controls_text = " | ← → seek | Space=pause"
        self.info_var.set(f"{basename}{existing_text}{controls_text}")

        labeled = self._count_labeled()
        total = len(self.videos)
        remaining = total - labeled
        # current position among all videos (1-based)
        pos = self.index + 1
        self.progress_var.set(
            f"Film {pos}/{total}  •  oznaczonych: {labeled}  •  pozostało: {remaining}"
        )

    # ------------------------------------------------------------------
    # Slider callbacks
    # ------------------------------------------------------------------

    def on_slider_press(self, event: tk.Event) -> str:
        self._slider_dragging = True
        self._set_slider_from_x(event.x)
        return "break"

    def on_slider_drag(self, event: tk.Event) -> str:
        if self._slider_dragging:
            self._set_slider_from_x(event.x)
        return "break"

    def _set_slider_from_x(self, x_pos: int) -> None:
        width = max(1, self.timeline_scale.winfo_width())
        max_value = float(self.timeline_scale.cget("to"))
        click_x = min(max(x_pos, 0), width)
        target_ms = int((click_x / width) * max_value)
        self.slider_var.set(target_ms)
        self.update_timeline_label(target_ms, int(max_value))

    def on_slider_release(self, _event: object) -> None:
        self._slider_dragging = False
        self.seek_to_ms(int(self.slider_var.get()))

    def on_slider_change(self, value: str) -> None:
        if self._slider_dragging:
            self.update_timeline_label(int(float(value)))

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------

    def _load_current(self) -> None:
        """Queue a load command for videos[index]."""
        if self.index < 0 or self.index >= len(self.videos):
            return

        path = self.videos[self.index]
        self._loading = True
        self.current_path = path
        self._set_buttons_state("disabled")
        self.status_var.set("Ładowanie…")
        self.slider_var.set(0)
        self.timeline_scale.configure(to=1000)
        self.update_timeline_label(0, 0)
        self.update_info()

        def on_loaded() -> None:
            self._loading = False
            self.paused = False
            self.status_var.set("")
            self._set_buttons_state("normal")
            self.update_info()

        def on_error(exc: Exception) -> None:
            self._loading = False
            self.current_path = None
            self.status_var.set("")
            self._set_buttons_state("normal")
            messagebox.showerror("Playback error", str(exc))
            self.history.append(self.index)
            self._advance()

        self._vlc.submit(
            {
                "cmd": "load",
                "path": path,
                "attach_fn": self._attach_player_to_widget,
                "done": on_loaded,
                "error_fn": on_error,
            }
        )

    # ------------------------------------------------------------------
    # Labeling / navigation
    # ------------------------------------------------------------------

    def label_current(self, label: str) -> None:
        """
        Label the current video and save to CSV (including timestamp).
        Rewrites entire CSV to prevent duplicates.
        """
        if self.current_path is None:
            return

        filename = self.current_path.name
        self.labels[filename.lower()] = (filename, label, now_iso())
        write_labels(self.csv_path, self.labels)

    def _label_and_advance(self, label: str) -> None:
        if self._loading or self.current_path is None:
            return
        self.label_current(label)
        self.history.append(self.index)
        self._advance()

    def _skip(self) -> None:
        if self._loading:
            return
        self.history.append(self.index)
        self._advance()

    def _back(self) -> None:
        """
        Go back to the previous video in history.

        Note: _load_current will transparently skip it if it was labeled
        in the meantime (e.g. labeled, then Back pressed).
        """
        if self._loading or not self.history:
            return
        self.index = self.history.pop()
        self._load_current()

    def _advance(self) -> None:
        """
        Move to the next unlabeled video.

        KEY FIX: Uses _first_unlabeled_from to jump over any labeled videos
        that appear later in the list — whether labeled this session or before.
        """
        self.index += 1
        if self.index >= len(self.videos):
            self._vlc.submit({"cmd": "stop", "done": self._on_all_done})
            return
        self._load_current()

    def _on_all_done(self) -> None:
        labeled = self._count_labeled()
        messagebox.showinfo(
            "Gotowe",
            f"Wszystkie filmy oznaczone ({labeled}/{len(self.videos)}).\n"
            f"Etykiety zapisano w:\n{self.csv_path}",
        )
        self._quit()

    def _quit(self) -> None:
        self._vlc.submit({"cmd": "quit"})
        try:
            self.root.quit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Periodic tick  (timeline sync + loop/error detection)
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        if self.current_path is not None and not self._loading:
            self.sync_timeline()
            if not self.paused:
                try:
                    state = self._vlc.player.get_state()
                    if state == vlc.State.Ended:
                        self._vlc.submit(
                            {
                                "cmd": "load",
                                "path": self.current_path,
                                "attach_fn": self._attach_player_to_widget,
                                "done": lambda: None,
                                "error_fn": lambda e: None,
                            }
                        )
                    elif state == vlc.State.Error:
                        messagebox.showerror(
                            "Playback error",
                            f"VLC reported an error:\n{self.current_path}",
                        )
                        self.history.append(self.index)
                        self._advance()
                        return
                except Exception:
                    pass

        self.root.after(TICK_INTERVAL_MS, self._tick)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        self.root.after(0, self._load_current)
        self.root.after(TICK_INTERVAL_MS, self._tick)
        self.root.mainloop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Label videos with Tk + VLC playback")
    parser.add_argument("folder", help="Folder containing videos")
    parser.add_argument("--csv", default="labels.csv", help="Output labels CSV path")
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Remove existing labels for videos in FOLDER before starting",
    )
    parser.add_argument(
        "--width", type=int, default=960, help="Video area width in pixels"
    )
    parser.add_argument(
        "--height", type=int, default=540, help="Video area height in pixels"
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"Folder does not exist or is not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "label"])

    try:
        app = VideoLabelerApp(
            folder=folder,
            csv_path=csv_path,
            width=args.width,
            height=args.height,
            purge=args.purge,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    app.run()


if __name__ == "__main__":
    main()
