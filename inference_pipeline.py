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
    sam, generator, age_regressor, _, _ = load_models(args.models_dir)
    sam.to(device).eval()
    generator.to(device).eval()
    age_regressor.to(device).eval()

    # Load trained adapter
    adapter = LatentAdapter(
        n_styles=getattr(generator, "n_latent", 18),
        style_dim=getattr(generator, "style_dim", 512)
    )

    ckpt = torch.load(args.adapter, map_location=device)
    adapter.load_state_dict(ckpt.get("adapter_state", ckpt))
    adapter.to(device).eval()

    print(f"[DEBUG] Generator n_latent : {getattr(generator, 'n_latent', None)}")
    print(f"[DEBUG] Generator style_dim: {getattr(generator, 'style_dim', None)}")
    print(f"[DEBUG] Checkpoint keys    : {list(ckpt.keys())[:10]}")

    os.makedirs(args.output, exist_ok=True)

    # Map mode to age delta
    age_map = {
        "reg--": -40,
        "reg-":  -20,
        "pro+":   20,
        "pro++":  40
    }
    if args.mode not in age_map:
        raise ValueError(f"Unsupported mode: {args.mode}")
    age_delta = age_map[args.mode]

    # Stronger alpha scaling: 20y = 1 unit
    alpha = (age_delta / 20.0) * args.strength
    print(f"[INFO] mode={args.mode} -> age_delta={age_delta}y, alpha={alpha:.3f}")

    # ── FFHQ alignment ────────────────────────────────────────────────────────
    print(f"[INFO] Aligning: {args.input}")
    img = align_face(args.input, output_size=256)

    if img is None:
        print("[WARN] No face detected — falling back to unaligned image")
        img = Image.open(args.input).convert("RGB")
    else:
        base_name_preview = os.path.splitext(os.path.basename(args.input))[0]
        preview_path = os.path.join(args.output, f"{base_name_preview}_aligned.png")
        img.save(preview_path)
        print(f"[DEBUG] Aligned preview -> {preview_path}, size: {img.size}")

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
    print(f"[DEBUG] SAM input channels: {sam_in_channels}")

    if sam_in_channels == 4:
        sam_in = torch.cat([inp, torch.zeros_like(inp[:, :1])], dim=1)
    else:
        sam_in = inp

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        enc_out = sam(sam_in, return_latents=True)
        print(f"[DEBUG] enc_out type: {type(enc_out)}")

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
        print(f"[DEBUG] codes shape: {codes.shape}")

        # ── ADAPTER OUTPUT NOTE ───────────────────────────────────────────────
        # This assumes adapter.forward() returns a DELTA (not full codes).
        # If your adapter returns full codes, change to:
        #   adapted = adapter(codes)
        #   delta = adapted - codes
        # Confirm by checking the last line of LatentAdapter.forward() in train_adapter.py
        adapter_out = adapter(codes)
        print(f"[DEBUG] adapter_out shape: {adapter_out.shape}")
        print(f"[DEBUG] adapter_out mean={adapter_out.mean().item():.4f}  std={adapter_out.std().item():.4f}")

        if adapter_out.shape == codes.shape:
            # If values are close to zero -> adapter outputs delta directly
            # If values are similar magnitude to codes -> adapter outputs full codes
            is_delta = adapter_out.abs().mean().item() < codes.abs().mean().item() * 0.5
            if is_delta:
                print("[DEBUG] Adapter output looks like DELTA (small magnitude)")
                delta = adapter_out
            else:
                print("[DEBUG] Adapter output looks like FULL CODES (large magnitude)")
                delta = adapter_out - codes
        else:
            delta = adapter_out

        final_codes = codes + alpha * delta
        final_codes = torch.clamp(final_codes, -10, 10)  # W+ safe range

        print(f"[DEBUG] delta       mean={delta.mean().item():.4f}  std={delta.std().item():.4f}")
        print(f"[DEBUG] final_codes mean={final_codes.mean().item():.4f}  std={final_codes.std().item():.4f}")

        out = generator([final_codes], input_is_latent=True)
        img_gen = out[0] if isinstance(out, tuple) else out
        img_gen = (img_gen.clamp(-1, 1) + 1) / 2.0

    # ── Save ──────────────────────────────────────────────────────────────────
    out_img = (img_gen[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

    base_name = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output, f"{base_name}_{args.mode}.png")
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
    print(f"[DONE] Saved -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Age Progression Inference")

    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--adapter",    type=str, required=True)
    parser.add_argument("--input",      type=str, required=True)
    parser.add_argument("--output",     type=str, default="output_images")
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["reg--", "reg-", "pro+", "pro++"],
                        help="reg-- (-40y)  reg- (-20y)  pro+ (+20y)  pro++ (+40y)")
    parser.add_argument("--strength",   type=float, default=1.0)

    args = parser.parse_args()
    infer(args)