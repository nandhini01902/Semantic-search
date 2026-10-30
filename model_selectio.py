import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

RECORDS_FILE = BASE_DIR / "data" / "medical_records 1.json"

MODELS = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2",
    "Clinical": None,
}


def main():
    with open(RECORDS_FILE, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError("No medical records found.")

    sample_texts = []

    for record in records[:5]:
        text = record.get("note_text")

        if isinstance(text, str) and text.strip():
            sample_texts.append(text.strip())

    if not sample_texts:
        raise ValueError("No valid medical note text was found.")

    print("\n========== EMBEDDING MODEL COMPARISON ==========")

    for name, model_id in MODELS.items():

        if model_id is None:
            print(f"\nSkipping {name}: no model ID configured.")
            continue

        print(f"\nModel: {name}")
        print("Model ID:", model_id)

        model = SentenceTransformer(model_id)

        start_time = time.perf_counter()

        embeddings = model.encode(
            sample_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        elapsed = time.perf_counter() - start_time

        print("Number of texts:", len(sample_texts))
        print("Embedding shape:", embeddings.shape)
        print("Embedding dimension:", embeddings.shape[1])
        print("Encoding time:", round(elapsed, 4), "seconds")

        del model

    print("\nModel comparison completed.")


if __name__ == "__main__":
    main()