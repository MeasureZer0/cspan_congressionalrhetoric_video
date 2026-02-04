import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --------- Globals to pass results from callback to main loop ----------
latest_result = None
latest_timestamp_ms = -1


def result_callback(
    result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    global latest_result, latest_timestamp_ms
    latest_result = result
    latest_timestamp_ms = timestamp_ms


def main():
    global latest_result

    # --- OpenCV camera ---
    cap = cv2.VideoCapture(0)
    # Windows fallback (uncomment if needed):
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try index 1/2 or check permissions.")

    # --- MediaPipe Tasks: PoseLandmarker in LIVE_STREAM mode ---
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=result_callback,
        # Optional tuning:
        # num_poses=1,
        # output_segmentation_masks=False,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("Ignoring empty camera frame.")
                continue

            # Convert BGR -> RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Wrap as mp.Image (SRGB means RGB data)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Monotonic timestamps are important for LIVE_STREAM
            timestamp_ms = int(time.time() * 1000)

            # Async inference (callback will be invoked)
            landmarker.detect_async(mp_image, timestamp_ms)

            # ---- Draw latest landmarks (if any) on the *current* BGR frame ----
            if latest_result and latest_result.pose_landmarks:
                # Draw first detected pose
                for pose_landmarks in latest_result.pose_landmarks:
                    # pose_landmarks is a list of NormalizedLandmark
                    h, w, _ = frame_bgr.shape
                    for lm in pose_landmarks:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame_bgr, (cx, cy), 2, (0, 255, 0), -1)

            cv2.imshow("PoseLandmarker (MediaPipe Tasks)", cv2.flip(frame_bgr, 1))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC or q
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
