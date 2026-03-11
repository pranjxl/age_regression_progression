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
    def __init__(self, root, size=256, max_images=None):
        self.root = Path(root)
        self.files = sorted([
            p for p in self.root.rglob("*")
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ])
        if max_images is not None:
            self.files = self.files[:max_images]
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
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, w):
        B, S, D = w.shape
        x  = w.view(B * S, D)
        dx = self.mlp(x)
        return dx.view(B, S, D)  # DELTA only


# ── Helpers ───────────────────────────────────────────────────────────────────
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
    if sam_ch == 4:
        sam_in = torch.cat([imgs, torch.zeros_like(imgs[:, :1])], dim=1)
    else:
        sam_in = imgs
    enc_out = sam(sam_in, return_latents=True)
    return extract_codes(enc_out, batch_size, device)


def resume_from_checkpoint(ckpt_path, adapter, opt, scheduler, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    adapter.load_state_dict(ckpt["adapter_state"])
    if "opt_state" in ckpt:
        opt.load_state_dict(ckpt["opt_state"])
    if "sched_state" in ckpt:
        scheduler.load_state_dict(ckpt["sched_state"])
    start_epoch = ckpt.get("epoch", 1) - 1
    start_step  = ckpt.get("step", 0)
    print(f"[INFO] Resumed from {ckpt_path} | epoch={start_epoch+1} step={start_step}")
    return start_epoch, start_step


# ── Training ──────────────────────────────────────────────────────────────────
def train(args):
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    sam, generator, age_regressor, id_model, _ = load_models(args.models_dir)
    sam.to(device).eval()
    generator.to(device).eval()
    age_regressor.to(device).eval()
    id_model.to(device).eval()
    for m in [sam, generator, age_regressor, id_model]:
        for p in m.parameters():
            p.requires_grad_(False)

    try:
        sam_ch = sam.encoder.input_layer[0].in_channels
    except Exception:
        sam_ch = 3
    print(f"[INFO] SAM input channels: {sam_ch}")

    ds = ImageFolderDataset(args.data_dir, size=256, max_images=args.max_images)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)
    print(f"[INFO] Dataset: {len(ds)} images, {len(dl)} batches/epoch")

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

    w_mean = w_std = None
    if os.path.exists("w_mean.pt") and os.path.exists("w_std.pt"):
        w_mean = torch.load("w_mean.pt", map_location=device)
        w_std  = torch.load("w_std.pt",  map_location=device)
        print("[INFO] W+ manifold stats loaded")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    L_AGE      = 1.0
    L_ID       = 0.5
    L_CYCLE    = 0.3
    L_LPIPS    = args.lpips_w
    L_MANIFOLD = 0.01
    L_DELTA    = 0.001

    AGE_UNIT  = 20.0
    MAX_DELTA = 40.0

    step = 0  # global step counter across epochs

    if args.resume:
        _, step = resume_from_checkpoint(
            args.resume, adapter, opt, scheduler, device=device
        )

    try:
        for epoch in range(args.epochs):
            adapter.train()
            pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
            running = {"total": 0, "age": 0, "id": 0, "cycle": 0}
            epoch_step = 0

            for imgs, _ in pbar:
                step += 1
                epoch_step += 1
                imgs = imgs.to(device)
                B    = imgs.shape[0]

                with torch.no_grad():
                    codes = sam_encode(sam, imgs, sam_ch, B, device)

                delta_years = (torch.rand(B, device=device) * 2 - 1) * MAX_DELTA
                alpha       = (delta_years / AGE_UNIT).view(-1, 1, 1)

                delta  = adapter(codes)
                w_aged = (codes + alpha * delta).clamp(-5, 5)

                out_aged  = generator([w_aged], input_is_latent=True)
                imgs_aged = out_aged[0] if isinstance(out_aged, tuple) else out_aged

                imgs_aged_224 = F.interpolate(imgs_aged, size=224, mode="bilinear", align_corners=False)
                imgs_256      = F.interpolate(imgs_aged, size=256, mode="bilinear", align_corners=False)

                with torch.no_grad():
                    imgs_224_src = F.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
                    orig_age_out = age_regressor(imgs_224_src)
                    orig_age     = orig_age_out[0] if isinstance(orig_age_out, tuple) else orig_age_out
                    orig_age     = orig_age.squeeze(-1)
                pred_age_out = age_regressor(imgs_aged_224)
                pred_age     = pred_age_out[0] if isinstance(pred_age_out, tuple) else pred_age_out
                pred_age     = pred_age.squeeze(-1)
                target_age   = (orig_age + delta_years).clamp(0, 100)
                age_loss     = F.mse_loss(pred_age, target_age)

                imgs_112     = F.interpolate(imgs,      size=112, mode="bilinear", align_corners=False)
                imgs_aged_112 = F.interpolate(imgs_aged, size=112, mode="bilinear", align_corners=False)
                with torch.no_grad():
                    feat_orig = id_model(imgs_112)
                feat_aged = id_model(imgs_aged_112)
                id_loss   = F.mse_loss(feat_orig, feat_aged)

                with torch.no_grad():
                    imgs_aged_norm     = imgs_aged.clamp(-1, 1)
                    imgs_aged_256_norm = F.interpolate(imgs_aged_norm, size=256,
                                                       mode="bilinear", align_corners=False)
                    w_reencoded = sam_encode(sam, imgs_aged_256_norm, sam_ch, B, device)

                # Deterministic adapter for cycle step (disable dropout, keep grad)
                was_training = adapter.training
                adapter.eval()
                delta2      = adapter(w_reencoded)
                if was_training:
                    adapter.train()
                w_recovered = (w_reencoded + (-alpha) * delta2).clamp(-5, 5)
                cycle_loss  = F.mse_loss(w_recovered, codes)

                if w_mean is not None:
                    manifold_loss = (w_aged - w_mean).pow(2).mean()
                else:
                    manifold_loss = torch.tensor(0.0, device=device)

                delta_reg = delta.pow(2).mean()

                if lpips_fn is not None:
                    imgs_01      = (imgs + 1) / 2
                    imgs_aged_01 = (imgs_256.clamp(-1, 1) + 1) / 2
                    lpips_loss   = lpips_fn(imgs_aged_01 * 2 - 1, imgs_01 * 2 - 1).mean()
                else:
                    lpips_loss = torch.tensor(0.0, device=device)

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
                n = epoch_step
                pbar.set_postfix(
                    loss=f"{running['total']/n:.4f}",
                    age=f"{running['age']/n:.4f}",
                    id=f"{running['id']/n:.4f}",
                    cyc=f"{running['cycle']/n:.4f}",
                )

                if epoch_step % 2000 == 0:
                    from torchvision.utils import save_image
                    save_image((imgs_aged.clamp(-1,1)+1)/2,
                               f"/kaggle/working/sample_aged_ep{epoch+1}_step{epoch_step}.png", nrow=2)
                    save_image((imgs.clamp(-1,1)+1)/2,
                               f"/kaggle/working/sample_src_ep{epoch+1}_step{epoch_step}.png", nrow=2)

                if epoch_step % 5000 == 0:
                    mid_path = os.path.join(args.checkpoint_dir,
                                            f"adapter_ep{epoch+1}_step{epoch_step}.pt")
                    torch.save({
                        "epoch":         epoch + 1,
                        "step":          epoch_step,
                        "adapter_state": adapter.state_dict(),
                        "opt_state":     opt.state_dict(),
                        "sched_state":   scheduler.state_dict(),
                    }, mid_path)
                    print(f"\n[INFO] Mid-epoch checkpoint -> {mid_path}")

            ckpt_path = os.path.join(args.checkpoint_dir, f"adapter_epoch{epoch+1}.pt")
            torch.save({
                "epoch":         epoch + 1,
                "adapter_state": adapter.state_dict(),
                "opt_state":     opt.state_dict(),
                "sched_state":   scheduler.state_dict(),
            }, ckpt_path)
            print(f"[INFO] Saved -> {ckpt_path}")

    except KeyboardInterrupt:
        save_path = os.path.join(args.checkpoint_dir, f"adapter_interrupt_step{step}.pt")
        torch.save({
            "epoch":         epoch,
            "step":          step,
            "adapter_state": adapter.state_dict(),
            "opt_state":     opt.state_dict(),
            "sched_state":   scheduler.state_dict(),
        }, save_path)
        print(f"[INFO] Interrupted — checkpoint saved -> {save_path}")
        raise

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
    parser.add_argument("--resume",         type=str,   default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--max_images",     type=int,   default=None,
                        help="Limit dataset size e.g. 5000 for quick sanity run")
    args = parser.parse_args()
    train(args)