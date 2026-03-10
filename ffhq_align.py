import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

mp_face = mp.solutions.face_mesh

# MediaPipe FaceMesh indices for full eye contours
LEFT_EYE_IDX  = [33, 133, 160, 159, 158, 144, 153, 154, 155, 163]
RIGHT_EYE_IDX = [362, 385, 386, 387, 263, 373, 374, 380, 381, 382]


def _eye_center(landmarks, indices, w, h):
    pts = np.array([
        [landmarks.landmark[i].x * w, landmarks.landmark[i].y * h]
        for i in indices
    ])
    return pts.mean(axis=0)


def _umeyama(src, dst, estimate_scale=True):
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    A = np.dot(dst_demean.T, src_demean) / num
    U, S, Vt = np.linalg.svd(A)
    d = np.ones((dim,))
    if np.linalg.det(A) < 0:
        d[-1] = -1
    T = np.eye(dim + 1)
    T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), Vt))
    if estimate_scale:
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0
    T[:dim, :dim] *= scale
    T[:dim, dim] = dst_mean - np.dot(T[:dim, :dim], src_mean)
    return T


def align_face(image_path, output_size=256):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    with mp_face.FaceMesh(static_image_mode=True,
                          refine_landmarks=True,
                          min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0]

        # 5-point landmarks: averaged eye centers + nose + mouth corners
        left_eye    = _eye_center(lm, LEFT_EYE_IDX,  w, h)
        right_eye   = _eye_center(lm, RIGHT_EYE_IDX, w, h)
        nose        = np.array([lm.landmark[1].x * w,   lm.landmark[1].y * h])
        left_mouth  = np.array([lm.landmark[61].x * w,  lm.landmark[61].y * h])
        right_mouth = np.array([lm.landmark[291].x * w, lm.landmark[291].y * h])

        src = np.array([left_eye, right_eye, nose, left_mouth, right_mouth],
                       dtype=np.float32)

        # FFHQ canonical target positions (normalised to output_size)
        dst = np.array([
            [0.31 * output_size, 0.35 * output_size],  # left eye
            [0.69 * output_size, 0.35 * output_size],  # right eye
            [0.50 * output_size, 0.55 * output_size],  # nose
            [0.35 * output_size, 0.75 * output_size],  # left mouth
            [0.65 * output_size, 0.75 * output_size],  # right mouth
        ], dtype=np.float32)

        T = _umeyama(src, dst, estimate_scale=True)
        M = T[:2]  # 2x3 affine matrix

        aligned = cv2.warpAffine(
            img,
            M,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        aligned_rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        return Image.fromarray(aligned_rgb)