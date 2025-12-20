# utils.py
import os
import sys
import torch
from argparse import Namespace
import cv2

_THIS_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(_THIS_DIR, "models", "stylegan2_pytorch"))
sys.path.append(os.path.join(_THIS_DIR, "models", "stylegan2"))
sys.path.append(os.path.join(_THIS_DIR, "models", "stylegan2_ada"))

from models.psp import pSp
from models.encoders import IR_50
from models.dex_vgg import DEX_VGG


def load_face_detector(models_dir="models"):
    proto = os.path.join(models_dir, "deploy.prototxt")
    model = os.path.join(models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    if not os.path.exists(proto) or not os.path.exists(model):
        raise FileNotFoundError("Face detector model files not found in models/")
    return cv2.dnn.readNetFromCaffe(proto, model)


def crop_face(frame, box, padding=20):
    x1, y1, x2, y2 = box
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1 - padding), max(0, y1 - padding), min(W - 1, x2 + padding), min(H - 1, y2 + padding)
    return frame[y1:y2, x1:x2]


def load_models(models_dir: str):
    """
    Loads StyleGAN2 generator, SAM (MyTimeMachine), pSp encoder, Age regressor, and IR-SE50.
    Returns: (sam_model, generator, age_regressor, id_model, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    gen_ckpt = os.path.join(models_dir, "stylegan2-ffhq-config-f.pt")
    sam_ckpt = os.path.join(models_dir, "sam_ffhq_aging.pt")
    psp_ckpt = os.path.join(models_dir, "psp_ffhq_encode.pt")
    age_ckpt = os.path.join(models_dir, "dex_age_classifier.pth")
    id_ckpt = os.path.join(models_dir, "model_ir_se50.pth")

    # ----- Load StyleGAN2 generator -----
    from models.stylegan2_pytorch.model import Generator
    print("[INFO] Loading StyleGAN2 (Rosinality) FFHQ generator...")
    g = Generator(size=1024, style_dim=512, n_mlp=8)
    g_state = torch.load(gen_ckpt, map_location=device)
    if "g_ema" in g_state and isinstance(g_state["g_ema"], dict):
        g.load_state_dict(g_state["g_ema"], strict=False)
        print("[INFO] Loaded generator weights from g_ema.")
    elif "state_dict" in g_state:
        g.load_state_dict(g_state["state_dict"], strict=False)
    else:
        g.load_state_dict(g_state, strict=False)
    generator = g.to(device).eval()
    print("[INFO] StyleGAN2 generator loaded successfully!")

    # ----- Load SAM model (with embedded pSp encoder) -----
    print("[INFO] Loading SAM (MyTimeMachine) model...")
    psp_opts = Namespace(
        output_size=1024,
        checkpoint_path=sam_ckpt,
        stylegan_weights=gen_ckpt,
        device=device,
        start_from_latent_avg=False,
        start_from_encoded_w_plus=True,
        input_nc=4,
        pretrained_psp_path=psp_ckpt,
    )
    sam_model = pSp(psp_opts).to(device).eval()
    print("[INFO] SAM model loaded successfully!")

    # ----- Load Age Regressor -----
    print("[INFO] Loading Age Regressor...")
    age_regressor = DEX_VGG().to(device)
    if os.path.exists(age_ckpt):
        state = torch.load(age_ckpt, map_location=device)
        state = state.get("state_dict", state)
        state = {k[4:] if k.startswith("vgg.") else k: v for k, v in state.items()}
        age_regressor.load_state_dict(state, strict=False)
        age_regressor.eval()
        print("[INFO] Age Regressor loaded successfully!")
    else:
        print("[WARN] Age regressor checkpoint not found; continuing without it.")

    # ----- Load IR-SE50 -----
    print("[INFO] Loading Identity Model (IR-SE50)...")
    id_model = IR_50(112).to(device)
    if os.path.exists(id_ckpt):
        state = torch.load(id_ckpt, map_location=device)
        missing, unexpected = id_model.load_state_dict(state, strict=False)
        if unexpected:
            print(f"[WARN] IR-SE50 unexpected {len(unexpected)} keys (ignored).")
        id_model.eval()
        print("[INFO] IR-SE50 identity model loaded successfully!")
    else:
        print("[WARN] IR-SE50 checkpoint missing; ID loss will be disabled.")

    print("[INFO] All models loaded.\n")
    return sam_model, generator, age_regressor, id_model, device
