#!/usr/bin/env python3
# Download model weights necessary for the project.
import os

import requests
from colorama import Fore, init
from tqdm import tqdm

init(autoreset=True)


def download_file(url: str, save_path: str) -> None:
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Stream the download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with (
        open(save_path, "wb") as file,
        tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive chunks
                file.write(chunk)
                pbar.update(len(chunk))


if __name__ == "__main__":
    # Direct download
    urls = [
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ]
    # Compute a path relative to this script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(script_dir)
    save_dir = os.path.join(project_root, "data", "weights")

    save_paths = [os.path.join(save_dir, os.path.basename(url)) for url in urls]

    for url, save_path in zip(urls, save_paths, strict=True):
        download_file(url, save_path)
        print(Fore.GREEN + f"Downloaded to {os.path.abspath(save_path)}")
