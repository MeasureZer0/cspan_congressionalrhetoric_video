#!/usr/bin/env python3
"""
Video folder labeler:
- Plays videos from a folder
- Click buttons: Positive / Neutral / Negative
- Writes to labels.csv: filename,label
- Supports Skip, Back, Quit
"""

import argparse
import csv
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def list_videos(folder: Path) -> List[Path]:
    vids = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            vids.append(p)
    vids.sort(key=lambda x: x.name.lower())
    return vids


def read_labels(csv_path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Returns dict: filename -> (label, timestamp)
    filename stored as basename only to keep it stable if folder moves.
    """
    labels: Dict[str, Tuple[str, str]] = {}
    if not csv_path.exists():
        return labels
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return labels
        for row in reader:
            fn = (row.get("filename") or "").strip()
            lab = (row.get("label") or "").strip()
            ts = (row.get("timestamp") or "").strip()
            if fn and lab:
                labels[fn] = (lab, ts)
    return labels


def write_labels(csv_path: Path, labels: Dict[str, Tuple[str, str]]) -> None:
    """
    Writes labels to CSV deterministically in the required format:
    "filename","label"
    """
    tmp_path = csv_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "label"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for fn in sorted(labels.keys(), key=lambda s: s.lower()):
            lab, _ts = labels[fn]
            writer.writerow({"filename": fn, "label": lab})
    tmp_path.replace(csv_path)


class VideoLabelerApp:
    def __init__(
        self, folder: Path, csv_path: Path, fps_cap: float = 30.0, resize_w: int = 960
    ) -> None:
        self.folder = folder
        self.csv_path = csv_path
        self.fps_cap = max(1.0, fps_cap)
        self.resize_w = max(320, resize_w)

        self.videos = list_videos(folder)
        if not self.videos:
            raise RuntimeError(f"No videos found in {folder}")

        self.labels = read_labels(csv_path)

        self.root = tk.Tk()
        self.root.title("Video Labeler")

        # State
        self.index = 0
        self.history: List[int] = []
        self.current_cap: Optional[cv2.VideoCapture] = None
        self.current_path: Optional[Path] = None
        self.current_frame_tk: Optional[ImageTk.PhotoImage] = None
        self.paused = False
        self._pending_action: Optional[str] = (
            None  # 'positive'/'neutral'/'negative'/'skip'/'quit'/'back'
        )

        # UI
        self.video_label = ttk.Label(self.root)
        self.video_label.grid(row=0, column=0, columnspan=6, padx=10, pady=10)

        self.info_var = tk.StringVar(value="")
        self.info = ttk.Label(self.root, textvariable=self.info_var)
        self.info.grid(row=1, column=0, columnspan=6, padx=10, pady=(0, 8), sticky="w")

        self.btn_pos = ttk.Button(
            self.root,
            text="✅ Positive",
            command=lambda: self.set_action("positive"),
            padding="10 6",
        )
        self.btn_neu = ttk.Button(
            self.root,
            text="😐 Neutral",
            command=lambda: self.set_action("neutral"),
            padding="10 6",
        )
        self.btn_neg = ttk.Button(
            self.root,
            text="❌ Negative",
            command=lambda: self.set_action("negative"),
            padding="10 6",
        )
        self.btn_skip = ttk.Button(
            self.root,
            text="⏭ Skip",
            command=lambda: self.set_action("skip"),
            padding="10 6",
        )
        self.btn_back = ttk.Button(
            self.root,
            text="↩ Back",
            command=lambda: self.set_action("back"),
            padding="10 6",
        )
        self.btn_quit = ttk.Button(
            self.root,
            text="⏹ Quit",
            command=lambda: self.set_action("quit"),
            padding="10 6",
        )

        self.btn_pos.grid(row=2, column=0, padx=8, pady=10)
        self.btn_neu.grid(row=2, column=1, padx=8, pady=10)
        self.btn_neg.grid(row=2, column=2, padx=8, pady=10)
        self.btn_skip.grid(row=2, column=3, padx=8, pady=10)
        self.btn_back.grid(row=2, column=4, padx=8, pady=10)
        self.btn_quit.grid(row=2, column=5, padx=8, pady=10)

        # Keyboard shortcuts
        self.root.bind("<KeyPress-p>", lambda e: self.set_action("positive"))
        self.root.bind("<KeyPress-n>", lambda e: self.set_action("negative"))
        self.root.bind("<KeyPress-u>", lambda e: self.set_action("neutral"))
        self.root.bind("<KeyPress-s>", lambda e: self.set_action("skip"))
        self.root.bind("<KeyPress-b>", lambda e: self.set_action("back"))
        self.root.bind("<KeyPress-q>", lambda e: self.set_action("quit"))
        self.root.bind("<space>", lambda e: self.toggle_pause())

        self.root.protocol("WM_DELETE_WINDOW", lambda: self.set_action("quit"))

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def set_action(self, action: str) -> None:
        self._pending_action = action

    def open_video(self, path: Path) -> None:
        self.close_video()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        self.current_cap = cap
        self.current_path = path

    def close_video(self) -> None:
        if self.current_cap is not None:
            try:
                self.current_cap.release()
            except Exception:
                pass
        self.current_cap = None
        self.current_path = None

    def update_info(self) -> None:
        assert self.current_path is not None
        base = self.current_path.name
        existing = self.labels.get(base)
        existing_str = f" (already labeled: {existing[0]})" if existing else ""
        self.info_var.set(
            f"{self.index + 1}/{len(self.videos)}  |\
                    {base}{existing_str}  |  Space=pause"
        )

    def label_current(self, label: str) -> None:
        assert self.current_path is not None
        base = self.current_path.name
        self.labels[base] = (label, now_iso())
        write_labels(self.csv_path, self.labels)

    def go_next(self) -> None:
        self.index += 1
        if self.index >= len(self.videos):
            messagebox.showinfo(
                "Done", f"Reached end. Labels saved to:\n{self.csv_path}"
            )
            self.set_action("quit")

    def go_back(self) -> None:
        if not self.history:
            return
        self.index = self.history.pop()

    def handle_action(self) -> bool:
        """
        Executes any pending GUI action.
        Returns True if we should reload a (new) video afterwards.
        """
        if not self._pending_action:
            return False

        action = self._pending_action
        self._pending_action = None

        if action in ("positive", "neutral", "negative"):
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

    def show_frame(self, frame_bgr: np.ndarray) -> None:
        # Resize to fixed width
        h, w = frame_bgr.shape[:2]
        new_w = self.resize_w
        new_h = int(h * (new_w / w))
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert to Tk image
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.current_frame_tk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self.current_frame_tk)

    def run(self) -> None:
        # Main loop: load each video, play until action taken
        while True:
            if self.index < 0 or self.index >= len(self.videos):
                break

            current = self.videos[self.index]
            try:
                self.open_video(current)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.history.append(self.index)
                self.go_next()
                continue

            self.update_info()

            # Play this video (looping) until user action changes index/quit
            last_tick = 0.0
            frame_delay = 1.0 / self.fps_cap

            while True:
                self.root.update_idletasks()
                self.root.update()

                # If user clicked something, handle it
                if self.handle_action():
                    break  # reload next/prev video

                if self.current_cap is None:
                    return

                if self.paused:
                    time.sleep(0.02)
                    continue

                now = time.time()
                if now - last_tick < frame_delay:
                    time.sleep(0.001)
                    continue
                last_tick = now

                ok, frame = self.current_cap.read()
                if not ok:
                    # End of video: loop back to start
                    self.current_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                self.show_frame(frame)

            # next/prev will continue outer loop

        self.close_video()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing videos")
    parser.add_argument(
        "--csv",
        default="labels.csv",
        help="Output labels CSV path (default: labels.csv)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback FPS cap for display (default: 30)",
    )
    parser.add_argument(
        "--width", type=int, default=960, help="Display width in px (default: 960)"
    )
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Folder does not exist or is not a directory: {folder}", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv).expanduser().resolve()

    app = VideoLabelerApp(
        folder=folder, csv_path=csv_path, fps_cap=args.fps, resize_w=args.width
    )
    app.run()


if __name__ == "__main__":
    main()
