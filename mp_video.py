import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO_PATH = "data/data2024/595065321.mp4"
MODEL_PATH = "pose_landmarker_heavy.task"
OUTPUT_PATH = "output_pose.mp4"
FRAME_SKIP = 10

# --- Visibility / presence filtering thresholds ---
P_THRESH = 0.6  # presence: landmark exists / is in-frame
V_THRESH = 0.6  # visibility: landmark is likely visible (not occluded)
MIN_VISIBLE_LANDMARKS = 10  # skip drawing if fewer than this are "visible"


def landmark_is_visible(lm, p_thresh=P_THRESH, v_thresh=V_THRESH) -> bool:
    """
    Returns True if the landmark is confidently present+visible and inside normalized frame bounds.
    Uses getattr defaults because some builds may omit presence/visibility for certain outputs.
    """
    presence = getattr(lm, "presence", 1.0)
    visibility = getattr(lm, "visibility", 1.0)
    in_frame = (0.0 <= lm.x <= 1.0) and (0.0 <= lm.y <= 1.0)
    return in_frame and (presence >= p_thresh) and (visibility >= v_thresh)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        # Fallback to a reasonable default if FPS metadata is missing
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        # output_segmentation_masks=False,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                continue

            # OpenCV -> RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Monotonic timestamp for VIDEO mode
            timestamp_ms = int((frame_idx / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # ---- draw only "visible" landmarks ----
            if result.pose_landmarks:
                for pose_landmarks in result.pose_landmarks:
                    visible_lms = [
                        lm for lm in pose_landmarks if landmark_is_visible(lm)
                    ]

                    # Optional: skip drawing entirely if too few reliable landmarks
                    if len(visible_lms) < MIN_VISIBLE_LANDMARKS:
                        continue

                    for lm in visible_lms:
                        x = int(lm.x * width)
                        y = int(lm.y * height)
                        cv2.circle(frame_bgr, (x, y), 2, (0, 255, 0), -1)

            writer.write(frame_bgr)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved output to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
