from pathlib import Path

import json

import faiss
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from semantic_pipeline import OUTPUT_DIR, load_queries, load_records


TOP_K = 5


def main() -> None:
    record_vectors = np.load(OUTPUT_DIR / "medical_embeddings.npy").astype(np.float32)
    query_vectors = np.load(OUTPUT_DIR / "query_embeddings.npy").astype(np.float32)
    records = load_records()
    queries = load_queries()

    record_ids = [record["record_id"] for record in records]
    results = []

    for query_idx, query in enumerate(queries):
        q = query_vectors[query_idx].reshape(1, -1)
        cosine_scores = cosine_similarity(q, record_vectors)[0]
        cosine_top = np.argsort(cosine_scores)[::-1][:TOP_K]

        dot_scores = record_vectors @ q[0]
        dot_top = np.argsort(dot_scores)[::-1][:TOP_K]

        distance_scores = euclidean_distances(q, record_vectors)[0]
        distance_top = np.argsort(distance_scores)[:TOP_K]

        for rank, record_idx in enumerate(cosine_top, start=1):
            results.append({
                "query_index": query_idx,
                "query_number": query_idx + 1,
                "method": "cosine",
                "rank": rank,
                "record_index": int(record_idx),
                "record_id": record_ids[int(record_idx)],
                "score": float(cosine_scores[record_idx]),
            })

        for rank, record_idx in enumerate(dot_top, start=1):
            results.append({
                "query_index": query_idx,
                "query_number": query_idx + 1,
                "method": "dot_product",
                "rank": rank,
                "record_index": int(record_idx),
                "record_id": record_ids[int(record_idx)],
                "score": float(dot_scores[record_idx]),
            })

        for rank, record_idx in enumerate(distance_top, start=1):
            results.append({
                "query_index": query_idx,
                "query_number": query_idx + 1,
                "method": "euclidean_distance",
                "rank": rank,
                "record_index": int(record_idx),
                "record_id": record_ids[int(record_idx)],
                "score": float(distance_scores[record_idx]),
            })

    results_df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "retrieval_results_all_queries.csv"
    results_df.to_csv(output_path, index=False)
    print(f"Saved retrieval results to {output_path}")


if __name__ == "__main__":
    main()
