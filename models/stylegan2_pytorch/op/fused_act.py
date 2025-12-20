import torch
from torch import nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# CPU & GPU safe version of FusedLeakyReLU (no JIT compilation required)
# -----------------------------------------------------------------------------

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    """
    A safe fallback for StyleGAN2's fused bias + activation operation.
    Works on CPU without requiring fused CUDA kernels.
    """
    if bias is not None:
        input = input + bias.view(1, -1, 1, 1)
    return F.leaky_relu(input, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.register_parameter("bias", None)
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
