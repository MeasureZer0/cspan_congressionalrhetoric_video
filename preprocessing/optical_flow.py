import cv2
import numpy as np

# Initialize TV-L1 optical flow
OPTICAL_FLOW = cv2.optflow.DualTVL1OpticalFlow_create()
assert OPTICAL_FLOW is not None, "Failed to create TV-L1 Optical Flow instance."


def compute_tvl1_flow(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """
    Compute the optical flow between two frames using the TV-L1 algorithm.
    Args:
        prev_frame (np.ndarray): Previous frame in greyscale format.
        next_frame (np.ndarray): Next frame in greyscale format.
    """
    flow = OPTICAL_FLOW.calc(prev_frame, next_frame, None)  # (H, W, 2)
    return flow


def flow_to_image(flow: np.ndarray) -> np.ndarray:
    """
    Create a visual representation of optical flow. Does:
    1. Clamp flow_x and flow_y into [-40, 40]
    2. Normalize to [0, 255]
    3. Store as a 3-channel image [flow_x, flow_y, zeros]
    Args:
        flow (np.ndarray): Optical flow array of shape (H, W, 2).

    Returns:
        np.ndarray: Visual representation of optical flow as a 3-channel image.
    """
    flow = np.clip(flow, -40, 40)
    flow = ((flow + 40) * (255.0 / 80)).astype(np.uint8)

    # create 3-channel image with zero third channel
    h, w = flow.shape[:2]
    flow_img = np.zeros((h, w, 3), dtype=np.uint8)
    flow_img[..., 0] = flow[..., 0]  # horizontal (u)
    flow_img[..., 1] = flow[..., 1]  # vertical (v)
    flow_img[..., 2] = 0  # no effect channel
    return flow_img


def get_optical_flow_between_frames(
    prev_frame: np.ndarray, next_frame: np.ndarray
) -> np.ndarray:
    flow = compute_tvl1_flow(prev_frame, next_frame)
    flow_img = flow_to_image(flow)
    return flow_img


# cv2.imwrite(os.path.join(output_dir, f"flow_{idx:05d}.jpg"), flow_img)
