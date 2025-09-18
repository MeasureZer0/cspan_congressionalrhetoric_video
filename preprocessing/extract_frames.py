import cv2
from numpy import ndarray


def extract_frames(path: str, frame_skip: int = 10) -> list[ndarray]:
    """
    Extract frames from a VideoCapture object.

    Parameters
    ----------
    path : str
        Path to the video file.
    frame_skip: int, default=10
        Save only every N-th frame (sampling frequency).

    Returns
    -------
    frames : list
        A list of extracted frames (each frame as a NumPy array).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {path}; skipping.")
        return []

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
