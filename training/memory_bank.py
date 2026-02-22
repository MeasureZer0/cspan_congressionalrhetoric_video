import torch
import torch.nn.functional as F
from torch import nn


class MemoryBank(nn.Module):
    """
    Memory bank for self-supervised learning, \
        allowing for a larger number of negative samples.
    """

    bank: torch.Tensor
    ptr: torch.Tensor
    is_full: torch.Tensor

    def __init__(self, size: int, dim: int) -> None:
        super().__init__()
        self.size = size
        self.dim = dim

        init = F.normalize(torch.randn(size, dim), dim=1)

        self.register_buffer("bank", init)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("is_full", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def enqueue(self, z: torch.Tensor) -> None:
        """
        Adds new embeddings to the bank, replacing the oldest ones.
        """
        z = F.normalize(z.detach(), dim=1)

        batch_size = int(z.shape[0])
        ptr = int(self.ptr[0])

        if batch_size > self.size:
            z = z[-self.size :]
            batch_size = self.size

        if ptr + batch_size <= self.size:
            self.bank[ptr : ptr + batch_size] = z
        else:
            tail = self.size - ptr
            self.bank[ptr:] = z[:tail]
            self.bank[: batch_size - tail] = z[tail:]

        new_ptr = (ptr + batch_size) % self.size
        self.ptr[0] = new_ptr

        if new_ptr < ptr or (ptr + batch_size) >= self.size:
            self.is_full[0] = True

    def get(self) -> torch.Tensor:
        """
        Returns all valid embeddings currently stored in the bank.
        """
        if bool(self.is_full[0]):
            return self.bank.clone()
        return self.bank[: int(self.ptr[0])].clone()

    def __len__(self) -> int:
        return self.size if bool(self.is_full[0]) else int(self.ptr[0])

    def __repr__(self) -> str:
        return f"MemoryBank(size={self.size}, dim={self.dim}, \
                    filled={len(self)}/{self.size})"
