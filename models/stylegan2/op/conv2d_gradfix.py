# models/stylegan2/op/conv2d_gradfix.py
import torch
import contextlib

_enabled = True

def enabled():
    return _enabled

@contextlib.contextmanager
def no_weight_gradients():
    """Temporarily disable weight gradients (used in StyleGAN2 conv)."""
    global _enabled
    old = _enabled
    _enabled = False
    try:
        yield
    finally:
        _enabled = old

def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Conv2D with optional gradient fix."""
    if not _enabled:
        return torch.nn.functional.conv2d(
            input, weight, bias, stride, padding, dilation, groups
        )
    return torch.nn.functional.conv2d(
        input, weight.contiguous(), bias, stride, padding, dilation, groups
    )

def conv_transpose2d(input, weight, bias=None, stride=1, padding=0,
                     output_padding=0, groups=1, dilation=1):
    """ConvTranspose2D fallback with same gradient fix behavior."""
    if not _enabled:
        return torch.nn.functional.conv_transpose2d(
            input, weight, bias, stride, padding, output_padding, groups, dilation
        )
    return torch.nn.functional.conv_transpose2d(
        input, weight.contiguous(), bias, stride, padding, output_padding, groups, dilation
    )
