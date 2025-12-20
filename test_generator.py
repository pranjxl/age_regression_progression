# test_generator.py (run from your project root with env active)
import torch, numpy as np
from utils import load_models




encoder, generator, age_regressor, id_model, device = load_models("models")
print("device:", device)
n = getattr(generator, "n_latent", None)
print("generator.n_latent:", n)
latent_dim = 512
# create random latent
rand_wplus = torch.randn(1, n, latent_dim).to(device)
print("rand_wplus shape:", rand_wplus.shape, rand_wplus.dtype, rand_wplus.device)

with torch.no_grad():
    try:
        out = generator([rand_wplus], input_is_latent=True, return_latents=False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        print("generator produced:", out.shape, out.min().item(), out.max().item())
        print("Generator ok.")
    except Exception as e:
        print("Generator failed:", e)
        raise
