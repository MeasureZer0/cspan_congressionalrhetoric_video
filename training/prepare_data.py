import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

INPUT_IMAGES_DIR = Path("data/self-supervised")
OUTPUT_DIR = Path("data/pose_features")

model = YOLO('yolo11n-pose.pt')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def process_sequence(folder_path, k=0):
    images = sorted(list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png")))
    
    if not images:
        return None

    raw_sequence = []

    """
    Extracting raw data from images
    17 vectors of length 3 ([x, y, confidence])
    """
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        
        results = model(frame, verbose=False)
        
        current_vector = np.zeros(51)

        """
        Checking if the model found a human
        .data gives the vector
        """
        if results[0].keypoints is not None and results[0].keypoints.data.nelement() > 0:
            kpts = results[0].keypoints.data.cpu().numpy()[0]
            
            # Normalization of (x, y) coordinates
            h, w, _ = frame.shape
            kpts[:, 0] /= w # X
            kpts[:, 1] /= h # Y

            current_vector = kpts.flatten()
        
        raw_sequence.append(current_vector)

    if not raw_sequence:
        return None
    
    raw_sequence = np.array(raw_sequence)

    """
    Frame stacking for temporal context.
    Each vector contains absolute keypoint coordinates (x, y, conf)
    from the current frame concatenated with 'k' previous frames.
    Resulting vector length: 51 * (k + 1).
    """
    if k == 0:
        return raw_sequence

    augmented_sequence = []
    
    # Initialization
    history_buffer = [raw_sequence[0]] * k

    for i in range(len(raw_sequence)):
        current_frame = raw_sequence[i]
        
        window = [current_frame] + history_buffer
        
        stacked_vector = np.concatenate(window)
        augmented_sequence.append(stacked_vector)
        
        # Deleting the oldest frame, adding the newest one
        history_buffer.insert(0, current_frame)
        history_buffer.pop()

    return np.array(augmented_sequence)


folders = [f for f in INPUT_IMAGES_DIR.iterdir() if f.is_dir()]

for folder in tqdm(folders):
    pose_data = process_sequence(folder, k=2)
    
    if pose_data is not None:
        # Using .npy format
        save_path = OUTPUT_DIR / f"{folder.name}.npy"
        np.save(save_path, pose_data)