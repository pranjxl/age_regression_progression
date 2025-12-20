# test_models.py
from utils import load_models

if __name__ == "__main__":
    models_dir = "models"  # or your full path
    encoder, generator, age_regressor, id_model, device = load_models(models_dir)

    # Just confirm models are alive
    print("[TEST] Encoder:", type(encoder))
    print("[TEST] Generator:", type(generator))
    print("[TEST] Age Regressor:", type(age_regressor))
    print("[TEST] Identity Model:", type(id_model))
