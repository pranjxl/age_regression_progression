# finetune_generator.py
import os
import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

try:
    import lpips
    LPIPS_AVAILABLE = True
except Exception:
    LPIPS_AVAILABLE = False

from utils import load_models
from train_adapter import ImageFolderDataset, LatentAdapter

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device:", device)

    sam, generator, _, _, _ = load_models(args.models_dir)
    sam.to(device).eval()
    # Load adapter
    adapter = LatentAdapter(n_styles=getattr(generator, "n_latent", 18), style_dim=getattr(generator, "style_dim", 512), hidden=args.hidden).to(device)
    assert os.path.exists(args.adapter_ckpt), "Adapter checkpoint missing"
    adapter_ck = torch.load(args.adapter_ckpt, map_location=device)
    adapter.load_state_dict(adapter_ck["adapter_state"] if "adapter_state" in adapter_ck else adapter_ck)
    adapter.eval()
    for p in adapter.parameters():
        p.requires_grad_(False)

    # Now prepare generator for fine-tuning
    generator.to(device).train()
    # Unfreeze generator params
    for p in generator.parameters():
        p.requires_grad_(True)

    ds = ImageFolderDataset(args.data_dir, size=args.resolution)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # optimizer for generator only
    opt = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.9, 0.999))
    mse = nn.MSELoss()
    if LPIPS_AVAILABLE and args.use_lpips:
        lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    else:
        lpips_fn = None

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, _ in pbar:
            imgs = imgs.to(device)
            if sam.encoder.input_layer[0].in_channels == 4:
                zeros = torch.zeros_like(imgs[:, :1, :, :])
                sam_in = torch.cat([imgs, zeros], dim=1)
            else:
                sam_in = imgs

            with torch.no_grad():
                enc_out = sam(sam_in, return_latents=True)
                _, codes = enc_out if isinstance(enc_out, (list, tuple)) else (None, enc_out)
                if isinstance(codes, (list, tuple)):
                    codes = codes[0]
                if codes.shape[0] != imgs.shape[0]:
                    codes = codes[:imgs.shape[0]]

            adapted = adapter(codes.to(device))
            # generator forward
            try:
                out = generator([adapted], input_is_latent=True)
            except TypeError:
                out = generator([adapted], input_is_latent=True)
            img_gen = out[0] if isinstance(out, tuple) else out
            img_gen = (img_gen.clamp(-1, 1) + 1) / 2.0
            tgt = (imgs + 1) / 2.0 if imgs.min() < -0.5 else imgs
            if img_gen.shape != tgt.shape:
                tgt = torch.nn.functional.interpolate(tgt, size=img_gen.shape[2:], mode="bilinear", align_corners=False)

            loss = mse(img_gen, tgt)
            if lpips_fn is not None:
                loss = loss + args.lpips_w * lpips_fn(img_gen * 2 - 1, tgt * 2 - 1).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        # Save generator checkpoint each epoch
        torch.save({"epoch": epoch+1, "g_state": generator.state_dict()}, os.path.join(args.checkpoint_dir, f"generator_ft_epoch{epoch+1}.pt"))
        print(f"[INFO] Saved generator checkpoint epoch {epoch+1}")

    print("[DONE] Generator fine-tune finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--adapter_ckpt", required=True)
    parser.add_argument("--checkpoint_dir", default="checkpoints/generator")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--use_lpips", action="store_true")
    parser.add_argument("--lpips_w", type=float, default=0.8)
    args = parser.parse_args()
    train(args)
