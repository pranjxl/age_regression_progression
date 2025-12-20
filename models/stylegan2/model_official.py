# models/stylegan2/model_official.py
# -------------------------------------------------------------
# Final Compatible Wrapper for Rosinality-Style StyleGAN2 (FFHQ)
# Works with both generator versions (handles missing kwargs)
# -------------------------------------------------------------

import torch
from torch import nn

try:
    # Try to import the Rosinality generator
    from models.stylegan2_pytorch.model import Generator as RosinalityGenerator
except Exception as e:
    raise ImportError(
        "❌ Could not import Rosinality StyleGAN2 generator. "
        "Make sure 'models/stylegan2_pytorch/model.py' exists."
    ) from e


class Generator(nn.Module):
    """
    A compatibility wrapper for Rosinality-style StyleGAN2 generators.
    Silently ignores unsupported kwargs (e.g. randomize_noise) so it works with both.
    """

    def __init__(self, size=1024, style_dim=512, n_mlp=8):
        super().__init__()
        self.model = RosinalityGenerator(size=size, style_dim=style_dim, n_mlp=n_mlp)
        self.n_latent = getattr(self.model, "n_latent", 18)
        self.style_dim = style_dim
        self.size = size

    def forward(self, styles, input_is_latent=True, return_latents=False, **kwargs):
        """
        Forward through generator, dropping unsupported arguments automatically.
        """
        try:
            # Try to call with all kwargs (modern Rosinality version)
            return self.model(
                styles,
                input_is_latent=input_is_latent,
                return_latents=return_latents,
                **kwargs,
            )
        except TypeError as e:
            # Retry without any extra kwargs (for older versions)
            print("[WARN] Ignoring unsupported generator kwargs:", str(e))
            return self.model(
                styles,
                input_is_latent=input_is_latent,
                return_latents=return_latents,
            )
