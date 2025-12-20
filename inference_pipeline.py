# inference_pipeline.py
import os
import argparse
import torch
from PIL import Image
import cv2
from torchvision import transforms
from utils import load_models
from train_adapter import LatentAdapter


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

    # Prepare output directory
    os.makedirs(args.output, exist_ok=True)

    # Image preprocessing (MATCH TRAINING)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    img = Image.open(args.input).convert("RGB")
    inp = preprocess(img).unsqueeze(0).to(device)

    # SAM input handling (3ch vs 4ch)
    if sam.encoder.input_layer[0].in_channels == 4:
        zeros = torch.zeros_like(inp[:, :1])
        sam_in = torch.cat([inp, zeros], dim=1)
    else:
        sam_in = inp

    # -------- Inference --------
    with torch.no_grad():
        enc_out = sam(sam_in, return_latents=True)

        if isinstance(enc_out, (list, tuple)):
            _, codes = enc_out
            if isinstance(codes, (list, tuple)):
                codes = codes[0]
        else:
            codes = enc_out

        # Apply trained adapter (NO age_delta)
        adapted_codes = adapter(codes)

        out = generator([adapted_codes], input_is_latent=True)
        img_gen = out[0] if isinstance(out, tuple) else out
        img_gen = (img_gen.clamp(-1, 1) + 1) / 2.0

    # Save output
    out_img = (
        img_gen[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        * 255
    ).astype("uint8")

    out_path = os.path.join(args.output, "aged_output.png")
    cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))

    print(f"[DONE] Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Age Progression Inference")

    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--adapter", type=str, required=True, help="Path to trained adapter checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input face image")
    parser.add_argument("--output", type=str, default="output_images", help="Output directory")

    args = parser.parse_args()
    infer(args)
