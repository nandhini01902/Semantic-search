from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

EMBEDDINGS_FILE = OUTPUT_DIR / "medical_embeddings.npy"
PLOT_FILE = OUTPUT_DIR / "pca_visualization.png"

RANDOM_STATE = 42


def main():
    embeddings = np.load(EMBEDDINGS_FILE)

    if embeddings.shape[0] < 2 or embeddings.shape[1] < 2:
        raise ValueError(
            "At least two records and two embedding dimensions are required."
        )

    pca = PCA(
        n_components=2,
        random_state=RANDOM_STATE,
    )

    points = pca.fit_transform(embeddings)

    print("Original shape:", embeddings.shape)
    print("2D shape:", points.shape)

    print(
        "Variance explained by two components:",
        round(float(pca.explained_variance_ratio_.sum()), 4),
    )

    plt.figure(figsize=(10, 7))

    plt.scatter(
        points[:, 0],
        points[:, 1],
        alpha=0.6,
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA Visualization of Medical Embeddings")

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300)
    plt.close()

    print("Plot saved to:", PLOT_FILE)


if __name__ == "__main__":
    main()