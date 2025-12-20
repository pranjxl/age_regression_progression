import os
import sys
import cv2
import torch
import numpy as np
import logging
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=256, margin=40, post_process=True)

# local utils
from utils import load_models, load_face_detector, highlight_face, crop_face

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ----------------- Config -----------------
MODELS_DIR = "models"
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_FRAME_SKIP = 8
DEFAULT_ALPHA_DENOMINATOR = 50.0
PASP_RES = 512
DISPLAY_RES = 512
# ------------------------------------------


# Load models
encoder, generator, age_regressor, id_model, device = load_models(MODELS_DIR)
face_net = load_face_detector(MODELS_DIR)   # ✅ load OpenCV DNN face detector
logging.info("Device: %s", device)

GEN_N_LATENT = int(getattr(generator, "n_latent", 18))
logging.info("Generator expects %d latent layers (n_latent).", GEN_N_LATENT)

# Load age boundary
AGE_BOUNDARY_PATH = os.path.join(MODELS_DIR, "age_w_boundary.npy")
if not os.path.exists(AGE_BOUNDARY_PATH):
    logging.error("Age boundary file not found: %s", AGE_BOUNDARY_PATH)
    sys.exit(1)

age_boundary_np = np.load(AGE_BOUNDARY_PATH)
age_boundary = torch.from_numpy(age_boundary_np).float().to(device)
logging.info("Loaded age boundary with shape %s", tuple(age_boundary.shape))


# ---------- Transforms ----------
class SquarePad:
    def __call__(self, img):
        w, h = img.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        padding = (hp, vp, max_wh - w - hp, max_wh - h - vp)
        return TF.pad(img, padding, 0, "constant")


PASP_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),  # smaller for encoder stability
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


AGE_REGRESSOR_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_shown_shape_debug = False
# -------------------------------------------


# ---------- predict_age ----------
def predict_age(img_pil):
    """Predict age from a PIL image with multiple safe fallbacks."""
    tensor = AGE_REGRESSOR_TRANSFORM(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = age_regressor(tensor)

        if isinstance(output, dict):
            logits = output.get("fc8", None)
            if logits is None:
                for v in output.values():
                    if torch.is_tensor(v):
                        logits = v
                        break
        elif isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        if logits is None:
            return 50.0

        if logits.dim() == 0 or (logits.dim() == 1 and logits.numel() == 1):
            return float(logits.view(-1).item())
        if logits.dim() == 2 and logits.size(1) == 1:
            return float(logits.view(-1).item())
        if logits.dim() == 2 and logits.size(1) == 101:
            probs = torch.softmax(logits, dim=1)
            ages = torch.arange(0, 101, dtype=torch.float32, device=device)
            return float((probs * ages.unsqueeze(0)).sum(dim=1).item())
        return float(logits.mean().item())

def align_face_with_mtcnn(img_pil):
    """
    Align and crop face using MTCNN for StyleGAN2 encoder.
    Returns aligned PIL image or original if detection fails.
    """
    try:
        aligned = mtcnn(img_pil)
        if aligned is not None:
            aligned_img = transforms.ToPILImage()(aligned)
            return aligned_img
        else:
            logging.warning("MTCNN could not detect face; using original image.")
            return img_pil
    except Exception as e:
        logging.warning(f"Face alignment failed: {e}")
        return img_pil


def age_edit_image(img_pil, target_age, alpha_denominator=DEFAULT_ALPHA_DENOMINATOR):
    """
    Encode -> latent shift -> decode -> blend back to original -> return blended face + ages.
    """
    # Align with MTCNN (keep your original face shape for blending)
    aligned_pil = align_face_with_mtcnn(img_pil)
    if aligned_pil is None:
        aligned_pil = img_pil

    # Predict current age
    current_age = predict_age(aligned_pil)
    delta = float(target_age) - float(current_age)
    alpha = delta / float(alpha_denominator)
    alpha = max(min(alpha, 5.0), -5.0)

    # Encode image → W+
    img_tensor = PASP_TRANSFORM(aligned_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        enc_out = encoder(img_tensor, return_latents=True)
        if isinstance(enc_out, (tuple, list)):
            _, latents = enc_out
        else:
            latents = enc_out

        if latents.dim() == 2:
            latents = latents.unsqueeze(1).repeat(1, GEN_N_LATENT, 1)

        # Apply boundary
        boundary = age_boundary
        if boundary.dim() == 1:
            boundary = boundary.unsqueeze(0).unsqueeze(1).repeat(1, GEN_N_LATENT, 1)
        elif boundary.dim() == 2:
            boundary = boundary.unsqueeze(0).repeat(1, GEN_N_LATENT, 1)

        edited = latents + alpha * boundary

        # Decode
        out = generator([edited], input_is_latent=True, return_latents=False)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = (out.clamp(-1, 1) + 1) / 2.0
        out_img = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

    # --------- Blend generated face back to original ----------
    orig_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    h, w = orig_bgr.shape[:2]
    out_bgr_resized = cv2.resize(out_bgr, (w, h))

    # Create a smooth elliptical mask
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.45), int(h * 0.55))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (61, 61), 30)

    # Blend using seamlessClone if possible
    try:
        blended = cv2.seamlessClone(out_bgr_resized, orig_bgr, mask, center, cv2.NORMAL_CLONE)
    except Exception:
        alpha = mask.astype(np.float32) / 255.0
        blended = (alpha[..., None] * out_bgr_resized + (1 - alpha[..., None]) * orig_bgr).astype(np.uint8)

    if DISPLAY_RES:
        blended = cv2.resize(blended, (DISPLAY_RES, DISPLAY_RES))

    return blended, float(current_age), float(target_age)




# ---------- Webcam loop ----------
def run_webcam_mode(target_age, frame_skip=DEFAULT_FRAME_SKIP, conf_threshold=0.7):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return

    logging.info("Webcam started. Press 'q' to quit, 's' to save snapshot.")
    frame_count, cached_result = 0, None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            display_frame = frame.copy()

            if frame_count % frame_skip == 0:
                try:
                    annotated, boxes = highlight_face(face_net, frame, conf_threshold=conf_threshold)
                    if boxes:
                        # pick largest face
                        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                        face = crop_face(frame, (x1, y1, x2, y2), padding=20)

                        if face.size > 0:
                            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                            gan_face, current_age, _ = age_edit_image(face_pil, target_age)

                            # resize GAN face back to bbox size
                            gan_face_resized = cv2.resize(gan_face, (x2-x1, y2-y1))

                            # create a soft elliptical mask
                            mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
                            center = (mask.shape[1]//2, mask.shape[0]//2)
                            axes = (mask.shape[1]//2, mask.shape[0]//2)
                            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                            mask = cv2.GaussianBlur(mask, (25,25), 10)

                            # seamless clone into frame
                            try:
                                display_frame = cv2.seamlessClone(
                                    gan_face_resized, frame, mask, (x1+(x2-x1)//2, y1+(y2-y1)//2),
                                    cv2.NORMAL_CLONE
                                )
                            except Exception:
                                # fallback: alpha blend
                                mask_f = mask.astype(float)/255.0
                                for c in range(3):
                                    display_frame[y1:y2, x1:x2, c] = (
                                        gan_face_resized[..., c]*mask_f +
                                        frame[y1:y2, x1:x2, c]*(1-mask_f)
                                    ).astype(np.uint8)

                            cached_result = display_frame
                            overlay_text = f"Now: {int(current_age)} → Target: {target_age}"
                        else:
                            overlay_text = "No face"
                    else:
                        overlay_text = "No face"
                except Exception as e:
                    logging.exception("Frame processing failed: %s", e)
                    overlay_text = "Error"

            else:
                overlay_text = f"Target: {target_age}"
                if cached_result is not None:
                    display_frame = cached_result

            # Overlay text
            cv2.putText(display_frame, overlay_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)

            cv2.imshow("Age Editing (q=quit, s=save)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and cached_result is not None:
                save_path = os.path.join(OUTPUT_DIR, f"snapshot_{target_age}.png")
                cv2.imwrite(save_path, cached_result)
                logging.info("Saved snapshot to %s", save_path)
    finally:
        cap.release()
        cv2.destroyAllWindows()




def run_image_mode(target_age, save_prefix="aged", conf_threshold=0.7):
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        logging.error("No images found in %s", INPUT_DIR)
        return

    for img_name in files:
        img_path = os.path.join(INPUT_DIR, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            logging.error("Could not read %s", img_path)
            continue

        try:
            annotated, boxes = highlight_face(face_net, frame, conf_threshold=conf_threshold)
            if not boxes:
                logging.warning("No face detected in %s", img_name)
                continue

            # largest face
            x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
            face = crop_face(frame, (x1, y1, x2, y2), padding=20)

            if face.size == 0:
                logging.warning("Empty crop in %s", img_name)
                continue

            # GAN age edit
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            gan_face, current_age, _ = age_edit_image(face_pil, target_age)

            cv2.imshow("Raw GAN Face", cv2.cvtColor(gan_face, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


            # resize GAN face to match bbox
            gan_face_resized = cv2.resize(gan_face, (x2-x1, y2-y1))

            # elliptical mask
            mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
            center = (mask.shape[1]//2, mask.shape[0]//2)
            axes = (mask.shape[1]//2, mask.shape[0]//2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (25,25), 10)

            # seamless clone
            blended = cv2.seamlessClone(
                gan_face_resized, frame, mask, (x1+(x2-x1)//2, y1+(y2-y1)//2),
                cv2.NORMAL_CLONE
            )

            # overlay text
            overlay_text = f"Now: {int(current_age)} → Target: {target_age}"
            cv2.putText(blended, overlay_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2, cv2.LINE_AA)

            # show + save
            cv2.imshow("Age Editing Result", blended)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            out_path = os.path.join(OUTPUT_DIR, f"{save_prefix}_{target_age}_{img_name}")
            cv2.imwrite(out_path, blended)
            logging.info("Saved blended result to %s", out_path)

        except Exception as e:
            logging.exception("Failed processing %s: %s", img_name, e)


# ---------- Entry ----------
if __name__ == "__main__":
    print("[INFO] Script started...")
    mode = input("Choose mode (image/webcam) [webcam]: ").strip().lower() or "webcam"
    try:
        target_age = int(input("Enter target age (e.g., 10, 60, 90) [60]: ") or 60)
    except Exception:
        target_age = 60

    if mode == "image":
        run_image_mode(target_age)
    elif mode == "webcam":
        run_webcam_mode(target_age)
    else:
        print("[ERROR] Invalid mode.")
