# train_adapter.py
import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

from utils import load_models


# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageFolderDataset(Dataset):
    def __init__(self, root, size=256):
        self.root = Path(root)
        self.files = sorted([
            p for p in self.root.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1,
                                   saturation=0.05, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), str(self.files[idx])


# ── Adapter ───────────────────────────────────────────────────────────────────
class LatentAdapter(nn.Module):
    """
    Maps (B, n_styles, 512) -> (B, n_styles, 512) DELTA only.
    forward() returns residual delta — caller does: w_new = w + alpha * delta
    """
    def __init__(self, n_styles=18, style_dim=512, hidden=1024, dropout=0.1):
        super().__init__()
        self.n_styles = n_styles
        self.style_dim = style_dim
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, style_dim),
        )
        # Zero-init: adapter starts as identity (delta=0)
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, w):
        B, S, D = w.shape
        x  = w.view(B * S, D)
        dx = self.mlp(x)
        return dx.view(B, S, D)   # DELTA only


# ── Helper ────────────────────────────────────────────────────────────────────
def extract_codes(enc_out, batch_size, device):
    if isinstance(enc_out, (list, tuple)):
        codes = None
        for item in enc_out[::-1]:
            if isinstance(item, torch.Tensor) and item.dim() >= 2:
                codes = item
                break
        if codes is None:
            _, codes = enc_out
            if isinstance(codes, (list, tuple)):
                codes = codes[0]
    else:
        codes = enc_out
    codes = codes.to(device)
    if codes.shape[0] != batch_size:
        codes = codes[:batch_size]
    return codes


def sam_encode(sam, imgs, sam_ch, batch_size, device):
    """Encode images through SAM, handling 3ch/4ch input."""
    if sam_ch == 4:
        sam_in = torch.cat([imgs, torch.zeros_like(imgs[:, :1])], dim=1)
    else:
        sam_in = imgs
    enc_out = sam(sam_in, return_latents=True)
    return extract_codes(enc_out, batch_size, device)


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # Frozen pretrained models
    sam, generator, age_regressor, id_model, _ = load_models(args.models_dir)
    sam.to(device).eval()
    generator.to(device).eval()
    age_regressor.to(device).eval()
    id_model.to(device).eval()
    for m in [sam, generator, age_regressor, id_model]:
        for p in m.parameters():
            p.requires_grad_(False)

    # Detect SAM input channels once
    try:
        sam_ch = sam.encoder.input_layer[0].in_channels
    except Exception:
        sam_ch = 3
    print(f"[INFO] SAM input channels: {sam_ch}")

    # Dataset
    ds = ImageFolderDataset(args.data_dir, size=256)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    print(f"[INFO] Dataset: {len(ds)} images, {len(dl)} batches/epoch")

    # Adapter
    n_styles  = getattr(generator, "n_latent", 18)
    style_dim = getattr(generator, "style_dim", 512)
    adapter   = LatentAdapter(n_styles, style_dim, args.hidden, dropout=0.1).to(device)

    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(dl), eta_min=1e-6
    )

    lpips_fn = None
    if LPIPS_AVAILABLE and args.use_lpips:
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
        print("[INFO] LPIPS enabled")

    # W+ manifold stats (optional)
    w_mean = w_std = None
    if os.path.exists("w_mean.pt") and os.path.exists("w_std.pt"):
        w_mean = torch.load("w_mean.pt", map_location=device)
        w_std  = torch.load("w_std.pt",  map_location=device)
        print("[INFO] W+ manifold stats loaded")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Loss weights
    L_AGE      = 1.0
    L_ID       = 0.5
    L_CYCLE    = 1.0
    L_LPIPS    = args.lpips_w          # FIX 4: use CLI value, not hardcoded 0.1
    L_MANIFOLD = 0.01
    L_DELTA    = 0.001                 # NEW: delta magnitude penalty

    AGE_UNIT  = 20.0    # 1 adapter unit = 20 years (matches inference alpha)
    MAX_DELTA = 40.0    # sample in [-40, +40] years

    for epoch in range(args.epochs):
        adapter.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        running = {"total": 0, "age": 0, "id": 0, "cycle": 0}

        for imgs, _ in pbar:
            imgs = imgs.to(device)                          # [B,3,256,256] in [-1,1]
            B    = imgs.shape[0]

            # ── Encode source images ──────────────────────────────────────────
            with torch.no_grad():
                codes = sam_encode(sam, imgs, sam_ch, B, device)   # [B,18,512]

            # ── Symmetric age delta sampling ──────────────────────────────────
            delta_years = (torch.rand(B, device=device) * 2 - 1) * MAX_DELTA
            alpha       = (delta_years / AGE_UNIT).view(-1, 1, 1)  # [B,1,1]

            # ── Adapter: predict delta, apply residual edit ───────────────────
            delta  = adapter(codes)                                 # [B,18,512]
            w_aged = (codes + alpha * delta).clamp(-10, 10)

            # ── Generate aged image ───────────────────────────────────────────
            out_aged   = generator([w_aged], input_is_latent=True)
            imgs_aged  = out_aged[0] if isinstance(out_aged, tuple) else out_aged
                                                                    # [B,3,H,W] H may be 1024

            # FIX 2: resize to 224 for age regressor (DEX expects 224)
            imgs_aged_224 = F.interpolate(imgs_aged, size=224, mode="bilinear", align_corners=False)
            imgs_256      = F.interpolate(imgs_aged, size=256, mode="bilinear", align_corners=False)

            # ── Age loss ──────────────────────────────────────────────────────
            with torch.no_grad():
                imgs_224_src = F.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
                orig_age     = age_regressor(imgs_224_src)
            pred_age    = age_regressor(imgs_aged_224)
            target_age  = (orig_age + delta_years).clamp(0, 100)
            age_loss    = F.mse_loss(pred_age, target_age)

            # ── Identity loss (FIX: MSE on features, more stable than cosine) ─
            with torch.no_grad():
                feat_orig = id_model(imgs)
            feat_aged = id_model(imgs_256)
            id_loss   = F.mse_loss(feat_orig, feat_aged)

            # ── Cycle loss: re-encode aged image, reverse edit, recover codes ─
            # FIX 1: re-encode the generated image (perceptual cycle, not math cycle)
            with torch.no_grad():
                # aged image must be in [-1,1] for SAM
                imgs_aged_norm = imgs_aged.clamp(-1, 1)
                imgs_aged_256_norm = F.interpolate(imgs_aged_norm, size=256,
                                                   mode="bilinear", align_corners=False)
                w_reencoded = sam_encode(sam, imgs_aged_256_norm, sam_ch, B, device)

            delta2      = adapter(w_reencoded)
            w_recovered = (w_reencoded + (-alpha) * delta2).clamp(-10, 10)
            cycle_loss  = F.mse_loss(w_recovered, codes)

            # ── Manifold regularisation (FIX 3: mean-only, more stable) ──────
            if w_mean is not None:
                manifold_loss = (w_aged - w_mean).pow(2).mean()
            else:
                manifold_loss = torch.tensor(0.0, device=device)

            # ── Delta magnitude penalty (NEW) ─────────────────────────────────
            delta_reg = delta.pow(2).mean()

            # ── LPIPS (optional) ──────────────────────────────────────────────
            if lpips_fn is not None:
                imgs_01      = (imgs + 1) / 2
                imgs_aged_01 = (imgs_256.clamp(-1, 1) + 1) / 2
                lpips_loss   = lpips_fn(imgs_aged_01 * 2 - 1, imgs_01 * 2 - 1).mean()
            else:
                lpips_loss = torch.tensor(0.0, device=device)

            # ── Total loss ────────────────────────────────────────────────────
            loss = (L_AGE      * age_loss
                  + L_ID       * id_loss
                  + L_CYCLE    * cycle_loss
                  + L_MANIFOLD * manifold_loss
                  + L_DELTA    * delta_reg
                  + L_LPIPS    * lpips_loss)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            opt.step()
            scheduler.step()

            running["total"] += loss.item()
            running["age"]   += age_loss.item()
            running["id"]    += id_loss.item()
            running["cycle"] += cycle_loss.item()
            n = pbar.n + 1
            pbar.set_postfix(
                loss=f"{running['total']/n:.4f}",
                age=f"{running['age']/n:.4f}",
                id=f"{running['id']/n:.4f}",
                cyc=f"{running['cycle']/n:.4f}",
            )

        ckpt_path = os.path.join(args.checkpoint_dir, f"adapter_epoch{epoch+1}.pt")
        torch.save({"epoch": epoch + 1, "adapter_state": adapter.state_dict()}, ckpt_path)
        print(f"[INFO] Saved -> {ckpt_path}")

    print("[DONE] Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir",     default="models")
    parser.add_argument("--data_dir",       required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints/adapter")
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--hidden",         type=int,   default=1024)
    parser.add_argument("--use_lpips",      action="store_true")
    parser.add_argument("--lpips_w",        type=float, default=0.1)
    args = parser.parse_args()
    train(args)