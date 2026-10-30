import numpy as np
from pathlib import Path

from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
RECORD_EMBEDDINGS_FILE = OUTPUT_DIR / "medical_embeddings.npy"
QUERY_EMBEDDINGS_FILE = OUTPUT_DIR / "query_embeddings.npy"

record_vectors = np.load(RECORD_EMBEDDINGS_FILE)
query_vectors = np.load(QUERY_EMBEDDINGS_FILE)

if record_vectors.ndim != 2 or query_vectors.ndim != 2:
    raise ValueError("Record and query embeddings must both be 2D arrays.")

if record_vectors.shape[1] != query_vectors.shape[1]:
    raise ValueError(
        "Record and query embeddings must have the same number of dimensions."
    )

if len(query_vectors) == 0:
    raise ValueError("At least one query embedding is required.")

query = query_vectors[0:]


cosine_scores = cosine_similarity(
    query,
    record_vectors,
)[0]


dot_scores = record_vectors @ query[0]

# 3. Euclidean distance: lower is better.
distance_scores = euclidean_distances(
    query,
    record_vectors,
)[0]

top_k = min(5, len(record_vectors))

cosine_top = np.argsort(cosine_scores)[::-1][:top_k]
dot_top = np.argsort(dot_scores)[::-1][:top_k]
distance_top = np.argsort(distance_scores)[:top_k]

print("Cosine top records:", cosine_top)
print("Dot-product top records:", dot_top)
print("Euclidean top records:", distance_top)