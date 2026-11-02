import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from semantic_pipeline import OUTPUT_DIR


def main() -> None:
    record_vectors = np.load(OUTPUT_DIR / "medical_embeddings.npy").astype(np.float32)
    query_vectors = np.load(OUTPUT_DIR / "query_embeddings.npy").astype(np.float32)

    query = query_vectors[0].reshape(1, -1)
    cosine_scores = cosine_similarity(query, record_vectors)[0]
    dot_scores = record_vectors @ query[0]
    euclidean_scores = euclidean_distances(query, record_vectors)[0]

    top_k = min(5, len(record_vectors))
    cosine_top = np.argsort(cosine_scores)[::-1][:top_k]
    dot_top = np.argsort(dot_scores)[::-1][:top_k]
    distance_top = np.argsort(euclidean_scores)[:top_k]

    print("Cosine top records:", cosine_top)
    print("Dot-product top records:", dot_top)
    print("Euclidean top records:", distance_top)


if __name__ == "__main__":
    main()
