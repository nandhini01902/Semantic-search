from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

EMBEDDINGS_FILE = OUTPUT_DIR / "medical_embeddings.npy"
REDUCED_FILE = OUTPUT_DIR / "medical_embeddings_reduced.npy"
PCA_MODEL_FILE = OUTPUT_DIR / "pca_model.joblib"

RANDOM_STATE = 42


def main():
    if not EMBEDDINGS_FILE.is_file():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDINGS_FILE}. "
            "Run embed.py first to generate it."
        )

    embeddings = np.load(EMBEDDINGS_FILE)

    print("Original embedding shape:", embeddings.shape)

    n_samples, n_features = embeddings.shape

    if n_samples < 2:
        raise ValueError("PCA requires at least two records.")

    n_components = min(50, n_samples, n_features)

    pca = PCA(
        n_components=n_components,
        random_state=RANDOM_STATE,
    )

    reduced_embeddings = pca.fit_transform(embeddings)

    print("Reduced embedding shape:", reduced_embeddings.shape)

    explained_variance = pca.explained_variance_ratio_.sum()

    print(
        "Explained variance ratio:",
        round(float(explained_variance), 4),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.save(
        REDUCED_FILE,
        reduced_embeddings.astype(np.float32),
    )

    joblib.dump(pca, PCA_MODEL_FILE)

    print("Reduced embeddings saved to:", REDUCED_FILE)
    print("PCA model saved to:", PCA_MODEL_FILE)


if __name__ == "__main__":
    main()