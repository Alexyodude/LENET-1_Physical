"""LeNet-5 sized to match the physical hardware.

Channel counts (4, 12) and feature-map sizes (28, 24, 12, 8, 4) match spec section 2.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 4, kernel_size=5)         # 28 -> 24
        self.c3 = nn.Conv2d(4, 12, kernel_size=5)        # 12 -> 8
        self.fc = nn.Linear(12 * 4 * 4, 10)              # final classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.c1(x))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.c3(x))
        x = F.avg_pool2d(x, 2)
        x = x.flatten(1)
        return self.fc(x)

    def forward_with_activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return every visualizable activation, ReLU'd where appropriate."""
        l1 = x
        l2 = F.relu(self.c1(x))
        l3 = F.avg_pool2d(l2, 2)
        l4 = F.relu(self.c3(l3))
        l5 = F.avg_pool2d(l4, 2)
        l6 = self.fc(l5.flatten(1))
        return {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5, "L6": l6}
