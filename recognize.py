# recognize.py
import os
import cv2
import torch
import numpy as np
from utils import load_models
from torchvision import transforms
from PIL import Image

# Paths
MODELS_DIR = "models"
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load models
encoder, generator, age_regressor, id_model, device = load_models(MODELS_DIR)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])



def predict_age(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Could not open {img_path}: {e}")
        return None, None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = age_regressor(tensor)

    # --- Handle different output formats ---
    if isinstance(output, dict):
        logits = output.get("fc8", None)
    elif isinstance(output, (tuple, list)):
        # Take the last element (usually logits)
        logits = output[-1]
    else:
        logits = output

    if logits is None:
        print("[ERROR] Could not extract logits from model output")
        return img, None

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    print(f"[DEBUG] Final logits shape: {logits.shape}")

    # Softmax and expected value
    probs = torch.softmax(logits, dim=1)  # [1, 101]
    ages = torch.arange(0, 101).float().to(device)
    pred_age = (probs * ages.unsqueeze(0)).sum(dim=1).item()

    return img, pred_age




if __name__ == "__main__":
    files = os.listdir(INPUT_DIR)
    if not files:
        print(f"[ERROR] No images found in {INPUT_DIR}")
    else:
        for img_name in files:
            img_path = os.path.join(INPUT_DIR, img_name)

            try:
                img, age = predict_age(img_path)
            except Exception as e:
                print(f"[ERROR] Failed to process {img_name}: {e}")
                continue

            print(f"[RESULT] {img_name}: Predicted Age = {age:.2f}")

            # Convert PIL → OpenCV BGR
            result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            cv2.putText(result, f"Age: {int(age)}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            out_path = os.path.join(OUTPUT_DIR, f"age_{img_name}")
            cv2.imwrite(out_path, result)
            print(f"[SAVED] {out_path}")
