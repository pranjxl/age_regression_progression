import os
from PIL import Image
from tqdm import tqdm

SRC_DIR = "datasets/faces_256_small"
DST_DIR = "datasets/faces_256_small_resized"

os.makedirs(DST_DIR, exist_ok=True)

for fname in tqdm(os.listdir(SRC_DIR)):
    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
        img = Image.open(os.path.join(SRC_DIR, fname)).convert("RGB")
        img = img.resize((256, 256), Image.BILINEAR)
        img.save(os.path.join(DST_DIR, fname), quality=95)
