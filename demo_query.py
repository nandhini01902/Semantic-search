import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_pipeline import OUTPUT_DIR, faiss_index_from_embeddings, get_topk_ranked_ids, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the medical record corpus for a custom text query.")
    parser.add_argument("query", nargs="?", default="fever with productive cough", help="Text to search for.")
    parser.add_argument("--k", type=int, default=5, help="Number of records to return.")
    args = parser.parse_args()

    records = load_records()
    record_ids = [record["record_id"] for record in records]
    record_vectors = np.load(OUTPUT_DIR / "medical_embeddings.npy").astype(np.float32)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_vector = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

    index = faiss_index_from_embeddings(record_vectors, metric="cosine")
    results = get_topk_ranked_ids(index, query_vector, record_ids, args.k, metric="cosine")

    print(f"Top {args.k} matches for: {args.query}")
    for row in results:
        print(f"{row['rank']}. {row['record_id']} | score={row['score']:.4f}")


if __name__ == "__main__":
    main()
