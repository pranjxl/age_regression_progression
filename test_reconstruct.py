# test_reconstruct.py
import os
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from utils import load_models

# ====================================
# CONFIG
# ====================================
SAVE_DIR = "debug_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

TRUNCATION = 1.0
REFINE = True
REFINE_STEPS = 150
REFINE_LR = 0.05
INPUT_RES = 1024

# ====================================
# LOAD MODELS
# ====================================
sam_model, generator, age_regressor, id_model, device = load_models("models")
GEN_N = getattr(generator, "n_latent", 18)
print(f"GEN_N = {GEN_N}, device = {device}")

# ====================================
# LOAD IMAGE
# ====================================
img_dir = "input_images"
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not img_files:
    raise FileNotFoundError("No image found in input_images/")
img_path = os.path.join(img_dir, img_files[0])
print("Using image:", img_path)

prep = transforms.Compose([
    transforms.Resize((INPUT_RES, INPUT_RES)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
img = Image.open(img_path).convert("RGB")
img_tensor = prep(img).unsqueeze(0).to(device)
print("Input tensor:", img_tensor.shape)

# ====================================
# FORWARD THROUGH SAM (4-channel input)
# ====================================
print("[INFO] Encoding via SAM...")

# Add dummy mask as 4th channel
mask = torch.ones_like(img_tensor[:, :1, :, :])  # shape: (1,1,1024,1024)
sam_input = torch.cat([img_tensor, mask], dim=1)  # shape: (1,4,1024,1024)
print("SAM input shape:", sam_input.shape)

with torch.no_grad():
    codes = sam_model(sam_input, return_latents=True)
    if isinstance(codes, (list, tuple)):
        codes = codes[-1]  # last latent if multiple returned
print("Encoded latent shape:", codes.shape)

# --- Fix: take only first sample if SAM returns a batch of 16 ---
if codes.ndim == 3 and codes.shape[0] > 1:
    print(f"[INFO] SAM returned batch of {codes.shape[0]}; keeping first sample only.")
    codes = codes[:1]

# Ensure latent shape compatibility with generator
if codes.ndim == 2:
    codes = codes.unsqueeze(1).repeat(1, GEN_N, 1)
elif codes.ndim == 3 and codes.shape[1] != GEN_N:
    codes = torch.nn.functional.interpolate(codes.unsqueeze(0), size=(GEN_N, 512)).squeeze(0)

print("Adjusted latent shape:", codes.shape)


# ====================================
# GENERATE IMAGE
# ====================================
print("[INFO] Generating initial reconstruction...")
with torch.no_grad():
    img_gen, _ = generator([codes], input_is_latent=True, return_latents=True)
img_gen = (img_gen.clamp(-1, 1) + 1) / 2.0
first_out = (img_gen[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

# ====================================
# REFINEMENT (optional)
# ====================================
if REFINE:
    print("[INFO] Refining latent...")
    latent_opt = codes.clone().detach().requires_grad_(True).to(device)
    opt = torch.optim.Adam([latent_opt], lr=REFINE_LR)
    mse = torch.nn.MSELoss()

    use_lpips = False
    try:
        import lpips
        lpips_loss = lpips.LPIPS(net="vgg").to(device).eval()
        use_lpips = True
        print("[INFO] LPIPS loaded.")
    except Exception:
        print("[WARN] LPIPS not available.")

    target_img = (img_tensor + 1) / 2.0

    for step in range(1, REFINE_STEPS + 1):
        opt.zero_grad()
        img_pred, _ = generator([latent_opt], input_is_latent=True, return_latents=True)
        img_pred = (img_pred.clamp(-1, 1) + 1) / 2.0
        loss = mse(img_pred, target_img)
        if use_lpips:
            loss = loss + 0.8 * lpips_loss(img_pred * 2 - 1, target_img * 2 - 1).mean()
        loss.backward()
        opt.step()
        if step % 30 == 0 or step == REFINE_STEPS:
            print(f"  Step {step}/{REFINE_STEPS} - loss: {loss.item():.5f}")

    with torch.no_grad():
        img_gen, _ = generator([latent_opt], input_is_latent=True, return_latents=True)
    img_gen = (img_gen.clamp(-1, 1) + 1) / 2.0

# ====================================
# SAVE RESULTS
# ====================================
out_img = (img_gen[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
cv2.imwrite(os.path.join(SAVE_DIR, "recon_input.png"), cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(SAVE_DIR, "recon_output.png"), cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(
    os.path.join(SAVE_DIR, "input_vs_output.png"),
    np.hstack([
        cv2.resize(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), (512,512)),
        cv2.resize(cv2.cvtColor(first_out, cv2.COLOR_RGB2BGR), (512,512)),
        cv2.resize(cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR), (512,512))
    ])
)
print("✅ Saved outputs in", SAVE_DIR)
