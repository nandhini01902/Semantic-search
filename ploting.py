from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from semantic_pipeline import OUTPUT_DIR


def main() -> None:
    embeddings = np.load(OUTPUT_DIR / "medical_embeddings.npy")
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA visualization of medical embeddings")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_visualization.png", dpi=300)
    plt.close()
    print("Saved PCA plot to outputs/pca_visualization.png")


if __name__ == "__main__":
    main()
