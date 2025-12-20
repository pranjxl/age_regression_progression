# Age Regression & Progression Pipeline

This repository contains an experimental deep learning pipeline for **age regression and age progression of human faces** using latent-space manipulation.

The project is built around a **StyleGAN2 FFHQ generator**, a **SAM (pSp-style) encoder**, and a **trainable latent adapter** that learns age-related transformations while preserving identity.

---

## 🔧 Core Components

- **StyleGAN2 (FFHQ)**  
  High-quality face generator used as the backbone.

- **SAM Encoder (MyTimeMachine / pSp-based)**  
  Encodes real images into the StyleGAN latent space.

- **Latent Adapter**  
  A lightweight neural module trained to modify latent codes for age transformation.

- **Age Regressor & Identity Network (IR-SE50)**  
  Used during training for supervision and identity preservation.

---

## 📂 Project Structure

age_regression_progression/

├── train_adapter.py # Adapter training

├── inference_pipeline.py # Lightweight inference

├── test_reconstruct.py # High-quality reconstruction & refinement

├── utils.py # Model loading utilities

├── models/ # Pretrained model weights (not included)

├── checkpoints/ # Adapter checkpoints (ignored by git)

├── datasets/ # Training images (ignored by git)

├── input_images/ # Test inputs (ignored by git)

├── output_images/ # Generated outputs (ignored by git)

└── README.md


---

## 🚀 Current Status

- ✅ Adapter training works at **256×256**
- ⚠️ **1024×1024 inference is memory-heavy** (6 GB VRAM limitation)
- ⚠️ Ongoing work to stabilize high-resolution inference
- 🔬 Experimental, research-oriented project

---

## 🧪 Notes

- Datasets, checkpoints, and generated images are **intentionally excluded** from this repository.
- This project is **not a production-ready system**.
- Results depend heavily on GPU memory and image resolution.

---

## 📌 Future Work

- Memory-efficient 1024×1024 training
- Better age-conditioning in latent adapter
- Unified high-quality inference pipeline
- Quantitative evaluation of age shift

---

## ⚠️ Disclaimer

This repository is for **educational and research purposes only**.  
No datasets or pretrained weights are distributed.
