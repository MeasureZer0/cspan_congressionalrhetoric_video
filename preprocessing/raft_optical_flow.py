import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

weights = Raft_Large_Weights.DEFAULT
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"RAFT is using device: {device}")
model = raft_large(weights=weights, progress=False).to(device)
model = model.eval()

transforms = weights.transforms()


def get_optical_flow_between_frames(
    prev_frame: torch.Tensor, next_frame: torch.Tensor
) -> torch.Tensor:
    """
    Compute optical flow using RAFT between two frames.
    Returns a torch tensor (flow_x, flow_y).
    Args:
        prev_frame (torch.Tensor): Previous frame, RGB float32
        next_frame (torch.Tensor): Next frame, RGB float32
    Returns:
        np.ndarray: Optical flow float32 (dx, dy)
    """
    prev_frame = prev_frame.unsqueeze(0).to(device)
    next_frame = next_frame.unsqueeze(0).to(device)

    # RAFT prediction
    with torch.no_grad():
        flows = model(prev_frame, next_frame)
        return flows[-1].squeeze(0)
