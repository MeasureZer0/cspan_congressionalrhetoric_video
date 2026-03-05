import cv2
from numpy import ndarray


def extract_frames(
    path: str,
    frame_skip: int = 30,
    skip_start_ratio: float = 0.1,
    skip_end_ratio: float = 0.1,
) -> list[ndarray]:
    """
    Extract frames from a VideoCapture object.

    Parameters
    ----------
    path : str
        Path to the video file.
    frame_skip: int, default=30
        Save only every N-th frame.
    skip_start_ratio: float, default=0.1
        Ratio of frames to skip at the start of the video. Maximum 5 seconds \
            at 30 fps.
    skip_end_ratio: float, default=0.1
        Ratio of frames to skip at the end of the video. Maximum 5 seconds \
            at 30 fps.

    Returns
    -------
    frames : list
        A list of extracted frames.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {path}; skipping.")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_start = min(int(total_frames * skip_start_ratio), 5 * 30)
    skip_end = min(int(total_frames * skip_end_ratio), 5 * 30)

    frames = []  # List to store selected frames
    counter = 0  # Counter for the current frame index
    while True:
        # - cap.read() returns:
        #     ret: Boolean indicating if a frame was read successfully
        #     frame: the actual frame image
        ret, frame = cap.read()

        # Skip the first "skip_start" frames and last "skip_end" frames
        if counter < skip_start or counter >= total_frames - skip_end:
            counter += 1
            continue

        # If no frame is returned:
        # - End of the video has been reached, OR
        # - An error occurred while reading
        if not ret:
            break

        # Store only every N-th frame
        if counter % frame_skip == 0:
            frames.append(frame)

        # Debugging visualization:
        # Uncomment the code below to display video frames while processing
        # cv2.imshow("Video Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        # Increment frame counter
        counter += 1

    # Cleanup: release system resources after processing
    # - cap.release(): closes the video file
    # - cv2.destroyAllWindows():
    #   closes any OpenCV windows (uncomment during visualization)
    cap.release()
    # cv2.destroyAllWindows()

    # Print how many frames were captured
    print(f"Frames captured from {path}: {len(frames)}")
    return frames
