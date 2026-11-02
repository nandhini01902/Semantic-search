from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from semantic_pipeline import MODEL_CONFIGS, OUTPUT_DIR, encode_texts, load_queries, load_records


def main() -> None:
    records = load_records()
    queries = load_queries()

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in MODEL_CONFIGS:
        model = SentenceTransformer(cfg["model_id"])
        record_texts = [record["note_text"] for record in records]
        query_texts = [query["query_text"] for query in queries]

        record_embeddings = encode_texts(model, record_texts, normalize=True)
        query_embeddings = encode_texts(model, query_texts, normalize=True)

        np.save(output_dir / f"medical_embeddings_{cfg['name']}.npy", record_embeddings)
        np.save(output_dir / f"query_embeddings_{cfg['name']}.npy", query_embeddings)
        print(f"Saved {cfg['name']} embeddings with shape {record_embeddings.shape}")

    default_model = MODEL_CONFIGS[0]["name"]
    default_records = np.load(output_dir / f"medical_embeddings_{default_model}.npy")
    default_queries = np.load(output_dir / f"query_embeddings_{default_model}.npy")
    np.save(output_dir / "medical_embeddings.npy", default_records)
    np.save(output_dir / "query_embeddings.npy", default_queries)
    print(f"Created compatibility files for {default_model} at medical_embeddings.npy and query_embeddings.npy")


if __name__ == "__main__":
    main()
