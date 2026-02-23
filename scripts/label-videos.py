#!/usr/bin/env python3
"""Tk + VLC video labeler.

# Based heavily on https://github.com/oaubert/python-vlc/blob/master/examples/tkvlc.py
Reuses the working VLC/Tk embedding approach (including macOS NSView handling)
from the tkvlc example, but provides the labeling workflow from label-videos.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import tkinter as tk
from ctypes import c_void_p, cdll
from ctypes.util import find_library
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import vlc

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
IS_MAC = sys.platform.startswith("darwin")
IS_WIN = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")


# Minimal copy of the tkvlc macOS helper: VLC needs an NSView, not the raw Tk id.
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


def read_labels(csv_path: Path) -> Dict[str, Tuple[str, str]]:
    labels: Dict[str, Tuple[str, str]] = {}
    if not csv_path.exists():
        return labels

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return labels
        for row in reader:
            filename = (row.get("filename") or "").strip()
            label = (row.get("label") or "").strip()
            timestamp = (row.get("timestamp") or "").strip()
            if filename and label:
                labels[filename] = (label, timestamp)
    return labels


def write_labels(csv_path: Path, labels: Dict[str, Tuple[str, str]]) -> None:
    tmp_path = csv_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "label"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for filename in sorted(labels.keys(), key=str.lower):
            label, _timestamp = labels[filename]
            writer.writerow({"filename": filename, "label": label})
    tmp_path.replace(csv_path)


class VideoLabelerApp:
    def __init__(
        self, folder: Path, csv_path: Path, width: int = 960, height: int = 540
    ) -> None:
        self.folder = folder
        self.csv_path = csv_path
        self.videos = list_videos(folder)
        if not self.videos:
            raise RuntimeError(f"No videos found in {folder}")

        self.labels = read_labels(csv_path)
        self.index = 0
        self.history: List[int] = []
        self.current_path: Optional[Path] = None
        self.paused = False
        self.pending_action: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("Video Labeler (TkVLC)")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.video_window: Optional[tk.Toplevel] = None
        video_parent: tk.Misc = self.root
        if IS_MAC:
            # On macOS, VLC can paint over the whole Tk window when attached to NSView.
            # Use a dedicated top-level for video so controls remain visible in the
            # main window.
            self.root.title("Label Controls")
            self.video_window = tk.Toplevel(self.root)
            self.video_window.title("Video Labeler (Video)")
            self.video_window.geometry(f"{width + 20}x{height + 20}")
            self.video_window.columnconfigure(0, weight=1)
            self.video_window.rowconfigure(0, weight=1)
            self.video_window.protocol(
                "WM_DELETE_WINDOW", lambda: self.set_action("quit")
            )
            video_parent = self.video_window
        else:
            self.root.geometry(f"{max(700, width + 40)}x{height + 180}")

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
            row=1, column=0, padx=10, pady=(0, 8), sticky="w"
        )

        buttons = ttk.Frame(self.root)
        buttons.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        for col in range(6):
            buttons.columnconfigure(col, weight=1)

        ttk.Button(
            buttons, text="Positive [P]", command=lambda: self.set_action("positive")
        ).grid(row=0, column=0, padx=4, sticky="ew")
        ttk.Button(
            buttons, text="Neutral [U]", command=lambda: self.set_action("neutral")
        ).grid(row=0, column=1, padx=4, sticky="ew")
        ttk.Button(
            buttons, text="Negative [N]", command=lambda: self.set_action("negative")
        ).grid(row=0, column=2, padx=4, sticky="ew")
        ttk.Button(
            buttons, text="Skip [S]", command=lambda: self.set_action("skip")
        ).grid(row=0, column=3, padx=4, sticky="ew")
        ttk.Button(
            buttons, text="Back [B]", command=lambda: self.set_action("back")
        ).grid(row=0, column=4, padx=4, sticky="ew")
        ttk.Button(
            buttons, text="Quit [Q]", command=lambda: self.set_action("quit")
        ).grid(row=0, column=5, padx=4, sticky="ew")

        self.root.bind("<KeyPress-p>", lambda _e: self.set_action("positive"))
        self.root.bind("<KeyPress-u>", lambda _e: self.set_action("neutral"))
        self.root.bind("<KeyPress-n>", lambda _e: self.set_action("negative"))
        self.root.bind("<KeyPress-s>", lambda _e: self.set_action("skip"))
        self.root.bind("<KeyPress-b>", lambda _e: self.set_action("back"))
        self.root.bind("<KeyPress-q>", lambda _e: self.set_action("quit"))
        self.root.bind("<space>", lambda _e: self.toggle_pause())
        self.root.protocol("WM_DELETE_WINDOW", lambda: self.set_action("quit"))
        if self.video_window is not None:
            self.video_window.bind(
                "<KeyPress-p>", lambda _e: self.set_action("positive")
            )
            self.video_window.bind(
                "<KeyPress-u>", lambda _e: self.set_action("neutral")
            )
            self.video_window.bind(
                "<KeyPress-n>", lambda _e: self.set_action("negative")
            )
            self.video_window.bind("<KeyPress-s>", lambda _e: self.set_action("skip"))
            self.video_window.bind("<KeyPress-b>", lambda _e: self.set_action("back"))
            self.video_window.bind("<KeyPress-q>", lambda _e: self.set_action("quit"))
            self.video_window.bind("<space>", lambda _e: self.toggle_pause())

        self.root.update_idletasks()
        if self.video_window is not None:
            self.video_window.update_idletasks()
            # Make the controls feel like a floating palette on macOS.
            try:
                self.root.transient(self.video_window)
            except tk.TclError:
                pass
            try:
                self.root.attributes("-topmost", True)
            except tk.TclError:
                pass
            self.root.resizable(False, False)
            self.root.geometry("+40+40")

        self.vlc_instance = vlc.Instance()
        self.player: Any = self.vlc_instance.media_player_new()
        self._attach_player_to_widget()

    def _attach_player_to_widget(self) -> None:
        self.root.update_idletasks()
        widget_id = self.video_canvas.winfo_id()

        if IS_WIN:
            self.player.set_hwnd(widget_id)
            return

        if IS_MAC:
            nsview = _GET_NSVIEW(widget_id)
            if nsview:
                self.player.set_nsobject(nsview)
                return

        if IS_LINUX or not IS_WIN:
            self.player.set_xwindow(widget_id)

    def set_action(self, action: str) -> None:
        self.pending_action = action

    def toggle_pause(self) -> None:
        if self.current_path is None:
            return
        self.paused = not self.paused
        self.player.pause()

    def close_video(self) -> None:
        try:
            self.player.stop()
        except Exception:
            pass
        self.current_path = None

    def open_video(self, path: Path) -> None:
        self.close_video()
        self.current_path = path
        media = self.vlc_instance.media_new(str(path))
        self.player.set_media(media)
        self._attach_player_to_widget()
        if self.player.play() == -1:
            raise RuntimeError(f"Failed to open video: {path}")
        self.paused = False

    def update_info(self) -> None:
        if self.current_path is None:
            return
        basename = self.current_path.name
        existing = self.labels.get(basename)
        existing_text = f" | already labeled: {existing[0]}" if existing else ""
        self.info_var.set(
            f"{self.index + 1}/{len(self.videos)} \
                | {basename}{existing_text} | Space=pause"
        )

    def label_current(self, label: str) -> None:
        if self.current_path is None:
            return
        self.labels[self.current_path.name] = (label, now_iso())
        write_labels(self.csv_path, self.labels)

    def go_next(self) -> None:
        self.index += 1
        if self.index >= len(self.videos):
            messagebox.showinfo(
                "Done", f"Reached end. Labels saved to:\n{self.csv_path}"
            )
            self.set_action("quit")

    def go_back(self) -> None:
        if self.history:
            self.index = self.history.pop()

    def handle_action(self) -> bool:
        action = self.pending_action
        if not action:
            return False
        self.pending_action = None

        if action in {"positive", "neutral", "negative"}:
            self.label_current(action)
            self.history.append(self.index)
            self.go_next()
            return True

        if action == "skip":
            self.history.append(self.index)
            self.go_next()
            return True

        if action == "back":
            self.go_back()
            return True

        if action == "quit":
            self.close_video()
            self.root.quit()
            return False

        return False

    def run(self) -> None:
        try:
            while True:
                if self.index < 0 or self.index >= len(self.videos):
                    break

                current = self.videos[self.index]
                try:
                    self.open_video(current)
                except Exception as exc:
                    messagebox.showerror("Playback error", str(exc))
                    self.history.append(self.index)
                    self.go_next()
                    continue

                self.update_info()

                while True:
                    try:
                        self.root.update_idletasks()
                        self.root.update()
                        if self.video_window is not None:
                            self.video_window.update_idletasks()
                            self.video_window.update()
                    except tk.TclError:
                        self.set_action("quit")

                    if self.handle_action():
                        break

                    if self.current_path is None:
                        return

                    if self.paused:
                        time.sleep(0.02)
                        continue

                    state = self.player.get_state()
                    if state == vlc.State.Ended:
                        self.player.stop()
                        self.player.play()
                    elif state == vlc.State.Error:
                        raise RuntimeError(f"Playback error: {self.current_path}")

                    time.sleep(0.01)
        finally:
            self.close_video()
            try:
                self.player.release()
            except Exception:
                pass
            try:
                self.vlc_instance.release()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Label videos with Tk + VLC playback")
    parser.add_argument("folder", help="Folder containing videos")
    parser.add_argument("--csv", default="labels.csv", help="Output labels CSV path")
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
    app = VideoLabelerApp(
        folder=folder, csv_path=csv_path, width=args.width, height=args.height
    )
    app.run()


if __name__ == "__main__":
    main()
