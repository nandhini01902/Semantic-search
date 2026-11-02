import time

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_pipeline import MODEL_CONFIGS, OUTPUT_DIR


def main() -> None:
    print("\n========== EMBEDDING MODEL COMPARISON ==========")
    for model_cfg in MODEL_CONFIGS:
        model = SentenceTransformer(model_cfg["model_id"])
        sample_texts = [
            "shortness of breath and chest pain",
            "persistent cough with fever",
            "knee pain after exercise",
            "reduced appetite and fatigue",
            "swollen leg with tenderness",
        ]

        start = time.perf_counter()
        embeddings = model.encode(sample_texts, convert_to_numpy=True, normalize_embeddings=True)
        elapsed = time.perf_counter() - start

        print(f"\nModel: {model_cfg['name']}")
        print("Model ID:", model_cfg["model_id"])
        print("Embedding shape:", embeddings.shape)
        print("Embedding dimension:", embeddings.shape[1])
        print("Encoding time:", round(elapsed, 4), "seconds")

    print("\nModel comparison completed.")


if __name__ == "__main__":
    main()
