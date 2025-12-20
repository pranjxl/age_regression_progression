# diagnose_generator.py
import torch, os
from utils import load_models

MODELS_DIR = "models"
generator_ckpt_path = os.path.join(MODELS_DIR, "stylegan2-ffhq-config-f.pt")  # adjust if your name differs

print("Loading models (only generator will be inspected)...")
encoder, generator, age_regressor, id_model, device = load_models(MODELS_DIR)

print("Device:", device)
print("Generator class:", type(generator))
print("generator.n_latent (if present):", getattr(generator, "n_latent", None))
print("Generator attributes (partial):", [k for k in dir(generator) if k.startswith("conv") or k.startswith("to_rgb")][:20])

# 1) print generator.state_dict() shapes
print("\n--- generator.state_dict() keys & shapes ---")
g_state = generator.state_dict()
for k, v in list(g_state.items())[:200]:
    print(k, tuple(v.shape))

# 2) load checkpoint g_ema if present and print its keys/shapes
if os.path.exists(generator_ckpt_path):
    print("\nLoading checkpoint:", generator_ckpt_path)
    ckpt = torch.load(generator_ckpt_path, map_location="cpu")
    # common key names: 'g_ema', 'g', 'g_running'
    possible_keys = [k for k in ckpt.keys() if isinstance(ckpt[k], dict)]
    print("Top-level dict keys in checkpoint:", list(ckpt.keys())[:40])
    g_ema = None
    for cand in ("g_ema", "g", "generator", "netG", "g_running"):
        if cand in ckpt:
            g_ema = ckpt[cand]
            print("Using checkpoint key:", cand)
            break
    if g_ema is None:
        # maybe checkpoint stored state_dict under 'state_dict' etc
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            g_ema = ckpt["state_dict"]
            print("Using 'state_dict' inside checkpoint")
    if g_ema is None:
        print("No obvious generator dict found in checkpoint. Printing keys top-level instead.")
    else:
        print("\n--- checkpoint generator keys & shapes ---")
        for k, v in list(g_ema.items())[:400]:
            if hasattr(v, "shape"):
                print(k, tuple(v.shape))
            else:
                print(k, type(v))

    # 3) compare keys between generator and g_ema
    if g_ema is not None:
        print("\n--- comparing checkpoint keys vs generator keys (showing mismatches) ---")
        g_keys = set(g_state.keys())
        c_keys = set(g_ema.keys())
        only_in_model = sorted(list(g_keys - c_keys))[:200]
        only_in_ckpt = sorted(list(c_keys - g_keys))[:200]
        print("Keys only in generator implementation (sample):", only_in_model[:30])
        print("Keys only in checkpoint (sample):", only_in_ckpt[:30])

        # check shapes for shared keys
        print("\nChecking shared keys shape mismatches:")
        shared = sorted(list(g_keys & c_keys))
        mismatches = []
        for k in shared:
            a = g_state[k].shape
            b = g_ema[k].shape
            if a != b:
                mismatches.append((k, a, b))
        if not mismatches:
            print("No shared-key shape mismatches detected.")
        else:
            print("Shared-key shape mismatches (first 40):")
            for m in mismatches[:40]:
                print(m)
else:
    print("\nGenerator checkpoint file not found at", generator_ckpt_path)
    print("Please confirm path or rename your generator checkpoint into models/ and rerun this script.")
