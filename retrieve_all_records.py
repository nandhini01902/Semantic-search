
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

RECORDS_FILE = OUTPUT_DIR / "medical_embeddings.npy"
QUERIES_FILE = OUTPUT_DIR / "query_embeddings.npy"
MEDICAL_RECORDS_FILE = BASE_DIR / "data" / "medical_records 1.json"

RESULTS_FILE = OUTPUT_DIR / "retrieval_results_all_queries.csv"

TOP_K = 5


def main():
   
    if not RECORDS_FILE.is_file():
        raise FileNotFoundError(
            f"Record embeddings not found: {RECORDS_FILE}. "
            "Run embed.py first."
        )

    if not QUERIES_FILE.is_file():
        raise FileNotFoundError(
            f"Query embeddings not found: {QUERIES_FILE}. "
            "Run embed.py first."
        )

    record_vectors = np.load(RECORDS_FILE).astype(np.float32)
    query_vectors = np.load(QUERIES_FILE).astype(np.float32)

    print("Medical record embeddings:", record_vectors.shape)
    print("Query embeddings:", query_vectors.shape)


    if record_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError(
            "Record and query embeddings must have the same dimensions."
        )

    if not np.isfinite(record_vectors).all():
        raise ValueError("Record embeddings contain invalid values.")

    if not np.isfinite(query_vectors).all():
        raise ValueError("Query embeddings contain invalid values.")

    
    record_ids = None

    if MEDICAL_RECORDS_FILE.exists():
        with open(
            MEDICAL_RECORDS_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            medical_records = json.load(file)

        if isinstance(medical_records, list):
            if len(medical_records) == len(record_vectors):
                record_ids = [
                    record.get("record_id", idx)
                    for idx, record in enumerate(medical_records)
                ]
            else:
                print(
                    "Warning: JSON record count does not match "
                    "embedding count. Saving array indices instead."
                )
        else:
            print(
                "Warning: Medical records JSON is not a list. "
                "Saving array indices instead."
            )

    if record_ids is None:
        record_ids = list(range(len(record_vectors)))

    # Step 4: Decide how many records to retrieve per query
    top_k = min(TOP_K, len(record_vectors))

    results = []

    # Step 5: Process EVERY query
    for query_idx, query in enumerate(query_vectors):

        # Convert one query into a 2D array
        query = query.reshape(1, -1)

        # Method 1: Cosine similarity
        cosine_scores = cosine_similarity(
            query,
            record_vectors
        )[0]

        cosine_top = np.argsort(cosine_scores)[::-1][:top_k]

        # Method 2: Dot product
        dot_scores = record_vectors @ query[0]

        dot_top = np.argsort(dot_scores)[::-1][:top_k]

        # Method 3: Euclidean distance
        distance_scores = euclidean_distances(
            query,
            record_vectors
        )[0]

        distance_top = np.argsort(distance_scores)[:top_k]

        # Step 6: Save cosine results
        for rank, record_idx in enumerate(cosine_top, start=1):
            results.append({
                "query_index": query_idx,
                "query_number": query_idx + 1,
                "method": "cosine",
                "rank": rank,
                "record_index": int(record_idx),
                "record_id": record_ids[record_idx],
                "score": float(cosine_scores[record_idx]),
            })

        # Step 7: Save dot-product results
        for rank, record_idx in enumerate(dot_top, start=1):
            results.append({
                "query_index": query_idx,
                "query_number": query_idx + 1,
                "method": "dot_product",
                "rank": rank,
                "record_index": int(record_idx),
                "record_id": record_ids[record_idx],
                "score": float(dot_scores[record_idx]),
            })

        # Step 8: Save Euclidean-distance results
        for rank, record_idx in enumerate(distance_top, start=1):
            results.append({
                "query_index": query_idx,
                "query_number": query_idx + 1,
                "method": "euclidean_distance",
                "rank": rank,
                "record_index": int(record_idx),
                "record_id": record_ids[record_idx],
                "score": float(distance_scores[record_idx]),
            })

        # Show progress
        if (
            (query_idx + 1) % 25 == 0
            or query_idx + 1 == len(query_vectors)
        ):
            print(
                f"Processed {query_idx + 1} "
                f"of {len(query_vectors)} queries"
            )

    # Step 9: Save all results to CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
        encoding="utf-8"
    )

    # Step 10: Display a summary
    print("\nRetrieval completed successfully!")
    print("Total queries processed:", len(query_vectors))
    print("Total medical records:", len(record_vectors))
    print("Dimensions used:", record_vectors.shape[1])
    print("Top records per method:", top_k)
    print("Total result rows:", len(results_df))
    print("Results saved to:", RESULTS_FILE)

    # Show results for the first query
    print("\nExample: Top results for Query 1")

    first_query_results = results_df[
        results_df["query_index"] == 0
    ]

    for method in [
        "cosine",
        "dot_product",
        "euclidean_distance"
    ]:
        print(f"\n{method}:")

        method_results = first_query_results[
            first_query_results["method"] == method
        ]

        print(
            method_results[
                ["rank", "record_index", "record_id", "score"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
