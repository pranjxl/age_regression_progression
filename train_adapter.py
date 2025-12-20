# train_adapter.py
import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# try lpips
try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

from utils import load_models  # your patched utils which loads SAM, generator, etc.

# --------------------------
# Simple dataset for images
# --------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root, size=1024):
        self.root = Path(root)
        self.files = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), str(p)

# --------------------------
# Adapter model
# --------------------------
class LatentAdapter(nn.Module):
    """
    Maps (B, n_styles, 512) -> (B, n_styles, 512)
    Implemented as a small per-style MLP with residual.
    """
    def __init__(self, n_styles=18, style_dim=512, hidden=1024):
        super().__init__()
        self.n_styles = n_styles
        self.style_dim = style_dim
        # We'll share weights across styles using a single MLP applied per-style
        self.mlp = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, style_dim),
        )
    def forward(self, w):
        # w: (B, n_styles, 512)
        B, S, D = w.shape
        x = w.view(B*S, D)
        dx = self.mlp(x)
        out = (x + dx).view(B, S, D)
        return out

# --------------------------
# Training loop
# --------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    # load models (sam encoder, generator)
    sam, generator, _, _, _ = load_models(args.models_dir)
    sam.to(device).eval()
    generator.to(device).eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    ds = ImageFolderDataset(args.data_dir, size=args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # instantiate adapter
    n_styles = getattr(generator, "n_latent", 18)
    adapter = LatentAdapter(n_styles, style_dim=getattr(generator, "style_dim", 512), hidden=args.hidden).to(device)

    opt = torch.optim.Adam(adapter.parameters(), lr=args.lr, betas=(0.9, 0.999))
    mse = nn.MSELoss()

    if LPIPS_AVAILABLE and args.use_lpips:
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
        print("[INFO] LPIPS available and enabled.")
    else:
        lpips_fn = None
        if args.use_lpips:
            print("[WARN] LPIPS requested but not available; continuing without it.")

    start_epoch = 0
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_loss = 0.0
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            # SAM expects 4 channels in this setup; add a zero channel if input is 3-ch
            if sam.encoder.input_layer[0].in_channels == 4:
                # add channel (zeros)
                zeros = torch.zeros_like(imgs[:, :1, :, :])
                sam_in = torch.cat([imgs, zeros], dim=1)
            else:
                sam_in = imgs

            with torch.no_grad():
                enc_out = sam(sam_in, return_latents=True)
                _, codes = enc_out if isinstance(enc_out, (list, tuple)) else (None, enc_out)
                if isinstance(codes, (list, tuple)):
                    codes = codes[0]
                # If batch mismatch (some SAMs return multiple), keep first sample per input:
                # many SAM checkpoints sometimes return batch>1; ensure codes batch aligns with imgs
                if codes.shape[0] != imgs.shape[0]:
                    # try to trim/pick corresponding ones
                    codes = codes[:imgs.shape[0]]

            # adapter -> generate
            adapted = adapter(codes.to(device))
            # generator expects [codes] + input_is_latent True
            try:
                out = generator([adapted], input_is_latent=True)
            except TypeError:
                out = generator([adapted], input_is_latent=True)
            img_gen = out[0] if isinstance(out, tuple) else out
            img_gen = (img_gen.clamp(-1, 1) + 1) / 2.0
            tgt = (imgs + 1) / 2.0 if imgs.min() < -0.5 else imgs  # if imgs normalized [-1,1] vs [0,1]
            # ensure shapes match: generator returns (B,3,H,W)
            if img_gen.shape != tgt.shape:
                tgt = torch.nn.functional.interpolate(tgt, size=img_gen.shape[2:], mode="bilinear", align_corners=False)

            loss = mse(img_gen, tgt)

            if lpips_fn is not None:
                # lpips expects [-1,1]
                loss = loss + args.lpips_w * lpips_fn(img_gen * 2 - 1, tgt * 2 - 1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=running_loss / (pbar.n + 1e-8))

        # save checkpoint per epoch
        torch.save({"epoch": epoch + 1, "adapter_state": adapter.state_dict()}, os.path.join(args.checkpoint_dir, f"adapter_epoch{epoch+1}.pt"))
        print(f"[INFO] Saved adapter checkpoint epoch {epoch+1}")

    print("[DONE] Adapter training finished.")

# --------------------------
# CLI
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models", help="dir with sam/psp/stylegan ckpts")
    parser.add_argument("--data_dir", required=True, help="images folder (FFHQ/CelebA-HQ)")
    parser.add_argument("--checkpoint_dir", default="checkpoints/adapter")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--use_lpips", action="store_true")
    parser.add_argument("--lpips_w", type=float, default=0.8)
    args = parser.parse_args()
    train(args)
