import torch
import torch.nn.functional as F
from torch import nn


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy loss for contrastive learning.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss between two sets of embeddings.
        """
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


class NTXentLossWithMemoryBank(nn.Module):
    """
    SimCLR loss extended to use a memory bank for more negative samples.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor, memory_bank: "MemoryBank"
    ) -> torch.Tensor:
        """
        Calculates loss using in-batch samples and memory bank samples.
        """
        B = z1.size(0)
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)

        pos_sim = (z1 * z2).sum(dim=1, keepdim=True) / self.temperature

        inbatch_sim = torch.mm(z1, z2.T) / self.temperature
        diag_mask = torch.eye(B, device=z1.device, dtype=torch.bool)
        inbatch_sim = inbatch_sim.masked_fill(diag_mask, float("-inf"))

        bank = memory_bank.get()
        bank_sim = torch.mm(z1, bank.T) / self.temperature

        logits = torch.cat([pos_sim, inbatch_sim, bank_sim], dim=1)
        labels = torch.zeros(B, dtype=torch.long, device=z1.device)
        loss = F.cross_entropy(logits, labels)

        memory_bank.enqueue(z2)
        return loss
