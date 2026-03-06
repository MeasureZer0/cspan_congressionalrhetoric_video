import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy loss for contrastive learning."""

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        batch_size = z1.size(0)
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)

        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.mm(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float("-inf"))

        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels])

        return F.cross_entropy(sim_matrix, labels)
