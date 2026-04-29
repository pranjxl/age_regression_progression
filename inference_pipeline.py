# inference_pipeline.py
import os
import argparse
import torch
from PIL import Image
import cv2
from torchvision import transforms
from utils import load_models
from train_adapter import LatentAdapter
from ffhq_align import align_face


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load models
    sam, generator, _, _, _ = load_models(args.models_dir)
    sam.to(device).eval()
    generator.to(device).eval()

    # Load trained adapter
    adapter = LatentAdapter(
        n_styles=getattr(generator, "n_latent", 18),
        style_dim=getattr(generator, "style_dim", 512)
    )
    ckpt = torch.load(args.adapter, map_location=device)
    adapter.load_state_dict(ckpt.get("adapter_state", ckpt))
    adapter.to(device).eval()
    print(f"[INFO] Adapter loaded from {args.adapter}")

    os.makedirs(args.output, exist_ok=True)

    # Mode → alpha direction
    age_map = {
        "pro": +1.0,   # older
        "reg": -1.0,   # younger
    }
    if args.mode not in age_map:
        raise ValueError(f"Unsupported mode: {args.mode}. Choose: reg, pro")

    alpha = age_map[args.mode] * args.strength
    print(f"[INFO] mode={args.mode} -> alpha={alpha:.3f}, preserve_id={args.preserve_id}")

    # ── FFHQ alignment ────────────────────────────────────────────────────────
    base_name_preview = os.path.splitext(os.path.basename(args.input))[0]

    if args.no_align:
        # Input is already FFHQ-aligned — skip to prevent double-crop distortion
        print(f"[INFO] Skipping alignment (--no_align): {args.input}")
        img = Image.open(args.input).convert("RGB")
        preview_path = os.path.join(args.output, f"{base_name_preview}_input.png")
        img.save(preview_path)
        print(f"[INFO] Input preview saved -> {preview_path}")
    else:
        print(f"[INFO] Aligning: {args.input}")
        img = align_face(args.input, output_size=256)
        if img is None:
            print("[WARN] No face detected — falling back to unaligned image")
            img = Image.open(args.input).convert("RGB")
        else:
            preview_path = os.path.join(args.output, f"{base_name_preview}_aligned.png")
            img.save(preview_path)
            print(f"[INFO] Aligned preview saved -> {preview_path}")

    # ── Preprocessing ─────────────────────────────────────────────────────────
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    inp = preprocess(img).unsqueeze(0).to(device)

    # SAM 3ch vs 4ch
    try:
        sam_in_channels = sam.encoder.input_layer[0].in_channels
    except Exception:
        sam_in_channels = 3
    print(f"[INFO] SAM input channels: {sam_in_channels}")

    if sam_in_channels == 4:
        # 4th channel = current age of input photo, normalized to [0,1]
        # 0.0 = newborn, 1.0 = 100 years old
        # e.g. 30-year-old → 0.30, 50-year-old → 0.50
        age_norm = args.source_age / 100.0
        age_map  = torch.ones_like(inp[:, :1]) * age_norm
        sam_in   = torch.cat([inp, age_map], dim=1)
        print(f"[INFO] SAM age channel set to {age_norm:.2f} ({args.source_age}y)")
    else:
        sam_in = inp

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        # Encode
        enc_out = sam(sam_in, return_latents=True)
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
        n_styles = codes.shape[1]
        print(f"[DEBUG] codes shape: {codes.shape}, mean={codes.mean():.4f}")

        # Adapter predicts age delta
        delta = adapter(codes)
        print(f"[DEBUG] delta norm={delta.norm():.4f}, mean={delta.mean():.4f}")

        # ── Safety clamp for frozen generator ─────────────────────────────
        # SAM was trained with PTI (per-image generator fine-tuning) which
        # allows large deltas (norm ~28). Our frozen generator cannot survive
        # that magnitude — it explodes the latent space.
        # We scale delta down to a safe per-layer norm before applying.
        per_layer_norm = delta.norm(dim=-1, keepdim=True)          # [1,18,1]
        safe_norm      = per_layer_norm.clamp(max=args.delta_max)  # cap per layer
        delta          = delta * (safe_norm / (per_layer_norm + 1e-8))
        print(f"[DEBUG] delta norm after clamp={delta.norm():.4f}")

        # ── Layer mixing for identity preservation ─────────────────────────
        # preserve_id=0.0 → edit all layers equally (max age change, max squish)
        # preserve_id=1.0 → protect coarse layers (bones/structure), edit fine layers (skin/texture)
        # Coarse layers (0-3)  → bone structure, jaw, skull → PROTECT when preserve_id is high
        # Mid layers   (4-7)  → face shape                 → partially protect
        # Fine layers  (8-17) → skin, wrinkles, texture    → always edit (safe, no squish)
        layer_mask = torch.ones_like(codes)
        layer_mask[:, :4,  :] = 1.0 - args.preserve_id        # coarse: protect bones
        layer_mask[:, 4:8, :] = 1.0 - args.preserve_id * 0.5  # mid: partial protection
        layer_mask[:, 8:,  :] = 1.0                            # fine: always edit skin

        final_codes = codes + alpha * delta * layer_mask
        final_codes = torch.clamp(final_codes, -5, 5)
        print(f"[DEBUG] final_codes mean={final_codes.mean():.4f}")

        # Generate
        out = generator([final_codes], input_is_latent=True)
        img_gen = out[0] if isinstance(out, tuple) else out
        img_gen = (img_gen.clamp(-1, 1) + 1) / 2.0

    # ── Save ──────────────────────────────────────────────────────────────────
    out_img = (img_gen[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_path  = os.path.join(
        args.output,
        f"{base_name}_{args.mode}_s{args.strength}_id{args.preserve_id}.png"
    )
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"[DONE] Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Age Progression/Regression Inference")
    parser.add_argument("--models_dir",  type=str,   default="models")
    parser.add_argument("--adapter",     type=str,   required=True,
                        help="Path to adapter checkpoint (.pt)")
    parser.add_argument("--input",       type=str,   required=True,
                        help="Input image path")
    parser.add_argument("--output",      type=str,   default="output_images")
    parser.add_argument("--mode",        type=str,   required=True,
                        choices=["reg", "pro"],
                        help="reg = younger, pro = older")
    parser.add_argument("--source_age",  type=float, default=30.0,
                        help="Estimated age of person in input photo (default=30)")
    parser.add_argument("--strength",    type=float, default=1.0,
                        help="Edit strength (default=1.0, try 0.5-2.0)")
    parser.add_argument("--preserve_id", type=float, default=0.0,
                        help="Identity preservation 0.0=off 1.0=max (protects bone structure)")
    parser.add_argument("--delta_max",   type=float, default=3.0,
                        help="Max per-layer delta norm (default=3.0, safe for frozen generator)")
    parser.add_argument("--no_align",    action="store_true",
                        help="Skip FFHQ alignment (use if input is already cropped perfectly)")
    
    args = parser.parse_args()
    infer(args)
    