"""Video Preprocessing Module"""

import os  # OS for safe file and directory path operations

import cv2  # OpenCV for handling video files (reading frames, opening videos)
import numpy as np  # NumPy for storing frames as numpy arrays
import pandas as pd  # Pandas for reading and handling CSV files in table format

# Define paths to input data:
# - data_dir: directory containing videos and labels
# - label_file: CSV file with labels (video filename + label)
data_dir = "data/sample_video"
label_file = "data/sample_video/labels.csv"


def extract_frames(cap: cv2.VideoCapture, frame_skip: int = 5) -> list:
    """
    Extract frames from a VideoCapture object.

    Parameters
    ----------
    cap : cv2.VideoCapture
        OpenCV VideoCapture object for the video file.
    frame_skip : int, default=5
        Save only every N-th frame (sampling frequency).

    Returns
    -------
    frames : list
        A list of extracted frames (each frame as a NumPy array).
    """
    frames = []
    count = 0  # frame counter
    while True:
        # - cap.read() returns:
        #     ret: Boolean indicating if a frame was read successfully
        #     frame: the actual frame image
        ret, frame = cap.read()

        # If no frame is returned:
        # - End of the video has been reached, OR
        # - An error occurred while reading
        if not ret:
            print("End of video or error occurred.")
            break

        # Store only every N-th frame
        if count % frame_skip == 0:
            frames.append(frame)

        # Debugging visualization:
        # Uncomment the code below to display video frames while processing
        # cv2.imshow("Video Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Increment frame counter
        count += 1

    # Cleanup: release system resources after processing
    # - cap.release(): closes the video file
    # - cv2.destroyAllWindows():
    #   closes any OpenCV windows (uncomment during visualization)
    cap.release()
    # cv2.destroyAllWindows()

    # Print how many frames were captured
    print(f"Frames captured: {len(frames)}")
    return frames


def load_data(data_dir: str, label_file: str) -> tuple:
    """
    Load video data and corresponding labels.

    Parameters
    ----------
    data_dir : str
        Path to the directory with video files.
    label_file : str
        Path to the CSV file with labels.

    Returns
    -------
    X : list
        A list of frame sequences for each video.
        Each element is a list of frames (NumPy arrays).
    y : list
        A list of labels corresponding to the videos.
    """
    # Load the label file into a Pandas DataFrame
    data = pd.read_csv(label_file)

    X, y = [], []

    # Iterate through each row in the CSV file
    for i, row in data.iterrows():
        video_name = row.iloc[0]  # video filename from CSV
        label = row.iloc[1]  # video label from CSV

        # Construct the video path
        video_path = os.path.join(data_dir, video_name)

        # Open the video file using OpenCV's VideoCapture
        # VideoCapture creates a video object from which we can read frames
        cap = cv2.VideoCapture(video_path)

        # Verify that the video file was successfully opened
        # cap.isOpened() returns True if the video is ready to process
        # If it fails, print an error and exit the script
        if not cap.isOpened():
            print(f"Error: Could not open video file {i}.")
            continue

        print(f"Video {i} file opened successfully!")

        frames = extract_frames(cap)
        # Save data only if frames were successfully extracted
        if frames:
            X.append(frames)
            y.append(label)
    return X, y


# Load the dataset
X, y = load_data(data_dir, label_file)

# Convert to NumPy arrays
X = np.array(X, dtype=object)
y = np.array(y)

print(f"X shape: {X.shape}, y shape: {y.shape}")
