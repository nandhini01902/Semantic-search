from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from semantic_pipeline import OUTPUT_DIR


def main() -> None:
    if (OUTPUT_DIR / "medical_embeddings.npy").exists():
        embeddings = np.load(OUTPUT_DIR / "medical_embeddings.npy")
    else:
        embeddings = np.load(OUTPUT_DIR / "medical_embeddings_MiniLM.npy")

    if embeddings.ndim != 2:
        raise ValueError("Expected a 2D embedding matrix.")

    target_dim = min(128, embeddings.shape[1])
    if target_dim < 2:
        raise ValueError("Embedding dimension is too small for PCA.")

    pca = PCA(n_components=target_dim, random_state=42)
    reduced = pca.fit_transform(embeddings)

    np.save(OUTPUT_DIR / "medical_embeddings_reduced.npy", reduced.astype(np.float32))
    joblib.dump(pca, OUTPUT_DIR / "pca_model.joblib")
    print(f"Saved PCA reduced embeddings of shape {reduced.shape}")


if __name__ == "__main__":
    main()
