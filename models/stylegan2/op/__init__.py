import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- CPU Fallback for FusedLeakyReLU ----
class FusedLeakyReLU(nn.Module):
    def __init__(self, channels, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, x):
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1)
        return F.leaky_relu(x, negative_slope=self.negative_slope) * self.scale


def fused_leaky_relu(x, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        x = x + bias.view(1, -1, 1, 1)
    return F.leaky_relu(x, negative_slope=negative_slope) * scale


# ---- Import CPU upfirdn2d ----
from .upfirdn2d import upfirdn2d as upfirdn2d
