# models/stylegan2/op/upfirdn2d.py  (CPU-safe, pure PyTorch)
import torch
import torch.nn.functional as F

# This implementation follows the math of the CUDA op but uses only
# standard PyTorch tensor ops so it runs on CPU (and CUDA) without
# needing a custom extension.

def _ensure_tuple2(x):
    if isinstance(x, (list, tuple)):
        assert len(x) == 2, "Expected a pair (x, y)"
        return int(x[0]), int(x[1])
    x = int(x)
    return x, x

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """
    input:  NCHW tensor
    kernel: 2D tensor (ky, kx)
    up:     int or (up_x, up_y)
    down:   int or (down_x, down_y)
    pad:    (pad_x0, pad_x1)  (applied symmetrically in y as well)
    returns: NCHW tensor
    """
    up_x, up_y   = _ensure_tuple2(up)
    down_x, down_y = _ensure_tuple2(down)
    pad_x0, pad_x1 = pad
    pad_y0, pad_y1 = pad  # symmetric like the original wrapper

    assert input.ndim == 4, "input must be NCHW"
    assert kernel.ndim == 2, "kernel must be 2D (ky, kx)"

    N, C, in_h, in_w = input.shape
    k = kernel.to(dtype=input.dtype, device=input.device)
    k_h, k_w = k.shape

    # ---- Convert to NHWC with 'minor' = C on the last axis (matches native path)
    x = input.permute(0, 2, 3, 1).contiguous()         # N, H, W, C
    x = x.view(-1, in_h, in_w, C)                      # (N), H, W, C

    # ---- Insert zeros for upsample
    # view -> pad zeros between pixels -> view back
    x = x.view(-1, in_h, 1, in_w, 1, C)                # (N), H, 1, W, 1, C
    if up_x > 1 or up_y > 1:
        x = F.pad(x, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])  # pads dims: (..., W_pad, ..., H_pad)
    x = x.view(-1, in_h * up_y, in_w * up_x, C)        # (N), H*up_y, W*up_x, C

    # ---- Pad (positive only via F.pad, negative via slicing)
    if any(v != 0 for v in (pad_x0, pad_x1, pad_y0, pad_y1)):
        x = F.pad(
            x,
            [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)],
        )
        x = x[
            :,
            max(-pad_y0, 0): x.shape[1] - max(-pad_y1, 0),
            max(-pad_x0, 0): x.shape[2] - max(-pad_x1, 0),
            :,
        ]

    # ---- Convolution with flipped kernel (per channel)
    # Move channels to NCHW for conv2d, convolve each channel independently (groups=C)
    x = x.permute(0, 3, 1, 2).contiguous()             # (N), C, H, W

    # Prepare kernel as depthwise conv weights
    w = torch.flip(k, [0, 1]).view(1, 1, k_h, k_w)     # 1,1,kh,kw
    w = w.to(x)                                        # match dtype/device
    w = w.repeat(C, 1, 1, 1)                           # C,1,kh,kw (depthwise)
    x = F.conv2d(x, w, bias=None, stride=1, padding=0, groups=C)  # (N), C, H', W'

    # ---- Downsample by striding
    x = x[:, :, ::down_y, ::down_x]                    # (N), C, H_out, W_out

    # ---- Restore batch dimension and return to NCHW
    out_h, out_w = x.shape[-2], x.shape[-1]
    x = x.view(N, C, out_h, out_w)

    return x
