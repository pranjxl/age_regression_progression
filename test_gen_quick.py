# test_gen_quick.py
import torch
from utils import load_models
import torchvision.transforms as T
from PIL import Image
import numpy as np

encoder, generator, age_regressor, id_model, device = load_models("models")

print("device:", device)
n = getattr(generator, "n_latent", None)
print("generator.n_latent:", n)

# sample random W+ latent
latent_dim = 512
wplus = torch.randn(1, n, latent_dim).to(device)

with torch.no_grad():
    out = generator([wplus], input_is_latent=True, return_latents=False)
    if isinstance(out, (tuple, list)):
        out = out[0]
    print("generator output shape:", out.shape, "min/max:", out.min().item(), out.max().item())

    img = (out.clamp(-1,1) + 1)/2.0
    img = (img[0].permute(1,2,0).cpu().numpy() * 255).astype("uint8")
    from PIL import Image
    Image.fromarray(img).save("debug_gen_random.png")
    print("Saved debug_gen_random.png")
