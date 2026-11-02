from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import faiss
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
    {
        "name": "MiniLM",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "lightweight general-purpose sentence embedding baseline",
    },
    {
        "name": "MPNet",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "description": "higher-quality dense sentence embeddings",
    },
    {
        "name": "MultiQA",
        "model_id": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "description": "retrieval-focused sentence embedding model",
    },
]

RECORDS_PATH = DATA_DIR / "medical_records 1.json"
QUERIES_PATH = DATA_DIR / "queries 1.json"


def load_records(path: Path = RECORDS_PATH) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)

    if not isinstance(records, list):
        raise ValueError(f"Expected a list of records in {path}, got {type(records).__name__}.")

    cleaned: List[Dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Record at index {index} is not a JSON object.")

        record_id = str(record.get("record_id", f"MR_{index:04d}"))
        note_text = record.get("note_text")
        if not isinstance(note_text, str):
            continue

        cleaned_text = " ".join(note_text.split())
        if not cleaned_text:
            continue

        normalized_record = dict(record)
        normalized_record["record_id"] = record_id
        normalized_record["note_text"] = cleaned_text
        cleaned.append(normalized_record)

    if not cleaned:
        raise ValueError(f"No valid records with non-empty note_text were found in {path}.")

    return cleaned


def load_queries(path: Path = QUERIES_PATH) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        queries = json.load(fh)

    if not isinstance(queries, list):
        raise ValueError(f"Expected a list of queries in {path}, got {type(queries).__name__}.")

    cleaned: List[Dict[str, Any]] = []
    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise TypeError(f"Query at index {index} is not a JSON object.")

        query_id = str(query.get("query_id", f"Q_{index:03d}"))
        query_text = query.get("query_text")
        if not isinstance(query_text, str):
            continue

        cleaned_text = " ".join(query_text.split())
        if not cleaned_text:
            continue

        relevant = query.get("relevant_record_ids", [])
        if isinstance(relevant, list):
            relevant_ids = [str(item) for item in relevant]
        else:
            relevant_ids = []

        normalized_query = dict(query)
        normalized_query["query_id"] = query_id
        normalized_query["query_text"] = cleaned_text
        normalized_query["relevant_record_ids"] = relevant_ids
        cleaned.append(normalized_query)

    if not cleaned:
        raise ValueError(f"No valid queries with text were found in {path}.")

    return cleaned


def encode_texts(model: SentenceTransformer, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def get_dimensionality_settings(native_dim: int) -> List[int]:
    settings: List[int] = []
    for target in (native_dim, 128, 64, 32):
        if target <= 0:
            continue
        if target == native_dim or target < native_dim:
            if target not in settings:
                settings.append(target)
    if not settings:
        settings = [native_dim]
    if native_dim not in settings:
        settings.insert(0, native_dim)
    unique = []
    for value in settings:
        if value not in unique:
            unique.append(value)
    return unique[:3]


def fit_dim_reduction(embeddings: np.ndarray, target_dim: int) -> Tuple[PCA, np.ndarray, np.ndarray]:
    if target_dim >= embeddings.shape[1]:
        return None, embeddings.copy(), embeddings.copy()

    pca = PCA(n_components=target_dim, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return pca, reduced.astype(np.float32), reduced.astype(np.float32)


def faiss_index_from_embeddings(embeddings: np.ndarray, metric: str = "cosine") -> faiss.Index:
    dim = embeddings.shape[1]
    if metric == "cosine":
        normalized = embeddings.copy()
        faiss.normalize_L2(normalized)
        index = faiss.IndexFlatIP(dim)
        index.add(normalized)
        return index
    if metric == "dot":
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index
    if metric == "euclidean":
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index
    raise ValueError(f"Unsupported FAISS metric '{metric}'.")


def retrieve_with_faiss(index: faiss.Index, query_vector: np.ndarray, k: int, normalize_query: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if k <= 0:
        raise ValueError("k must be positive.")

    q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    if normalize_query:
        faiss.normalize_L2(q)

    if index.ntotal == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    distances, indices = index.search(q, min(k, index.ntotal))
    return indices[0], distances[0]


def get_topk_ranked_ids(index: faiss.Index, query_vector: np.ndarray, ids: Sequence[str], k: int, metric: str = "cosine") -> List[Dict[str, Any]]:
    normalize_query = metric == "cosine"
    idxs, scores = retrieve_with_faiss(index, query_vector, k, normalize_query=normalize_query)
    rows: List[Dict[str, Any]] = []
    for rank, rec_idx in enumerate(idxs, start=1):
        if rec_idx == -1:
            continue
        rows.append({
            "rank": rank,
            "record_id": ids[int(rec_idx)],
            "score": float(scores[rank - 1]),
        })
    return rows


def precision_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for record_id in top_k if record_id in relevant_ids)
    return hits / k


def recall_at_k(relevant_ids: set[str], retrieved_ids: Sequence[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for record_id in top_k if record_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(relevant_ids: set[str], retrieved_ids: Sequence[str]) -> float:
    if not relevant_ids:
        return 0.0
    for rank, record_id in enumerate(retrieved_ids, start=1):
        if record_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_query(query: Dict[str, Any], retrieved_ids: Sequence[str], k_values: Sequence[int]) -> Dict[str, float]:
    relevant = set(query.get("relevant_record_ids", []))
    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"Precision@{k}"] = precision_at_k(relevant, retrieved_ids, k)
        metrics[f"Recall@{k}"] = recall_at_k(relevant, retrieved_ids, k)
    metrics["MRR"] = mean_reciprocal_rank(relevant, retrieved_ids)
    return metrics


def summarize_experiment(model_name: str, model_dim: int, target_dim: int, records: List[Dict[str, Any]], queries: List[Dict[str, Any]], record_embeddings: np.ndarray, query_embeddings: np.ndarray, k_values: Sequence[int]) -> pd.DataFrame:
    record_ids = [record["record_id"] for record in records]
    metric_rows = []
    for query_idx, query in enumerate(queries):
        q_vector = query_embeddings[query_idx]
        index = faiss_index_from_embeddings(record_embeddings, metric="cosine")
        retrieved_indices, _ = retrieve_with_faiss(index, q_vector, max(k_values), normalize_query=True)
        retrieved_ids = [record_ids[int(item)] for item in retrieved_indices if item != -1]
        metrics = evaluate_query(query, retrieved_ids, k_values)
        row = {
            "model_name": model_name,
            "model_dim": model_dim,
            "reduction_dim": target_dim,
            "query_id": query["query_id"],
            "query_text": query["query_text"],
        }
        row.update(metrics)
        metric_rows.append(row)

    summary = pd.DataFrame(metric_rows)
    return summary


def run_experiments() -> pd.DataFrame:
    records = load_records()
    queries = load_queries()
    summaries = []

    for model_cfg in MODEL_CONFIGS:
        print(f"\n=== Loading model: {model_cfg['name']} ===")
        model = SentenceTransformer(model_cfg["model_id"])
        record_texts = [record["note_text"] for record in records]
        query_texts = [query["query_text"] for query in queries]

        record_embeddings = encode_texts(model, record_texts, normalize=True)
        query_embeddings = encode_texts(model, query_texts, normalize=True)

        np.save(OUTPUT_DIR / f"medical_embeddings_{model_cfg['name']}.npy", record_embeddings)
        np.save(OUTPUT_DIR / f"query_embeddings_{model_cfg['name']}.npy", query_embeddings)

        settings = get_dimensionality_settings(record_embeddings.shape[1])
        for target_dim in settings:
            if target_dim == record_embeddings.shape[1]:
                reduced_records = record_embeddings.copy()
                reduced_queries = query_embeddings.copy()
            else:
                pca = PCA(n_components=target_dim, random_state=42)
                reduced_records = pca.fit_transform(record_embeddings)
                reduced_queries = pca.transform(query_embeddings)
                joblib.dump(pca, OUTPUT_DIR / f"pca_model_{model_cfg['name']}_{target_dim}.joblib")

            np.save(OUTPUT_DIR / f"medical_embeddings_{model_cfg['name']}_{target_dim}.npy", reduced_records.astype(np.float32))
            np.save(OUTPUT_DIR / f"query_embeddings_{model_cfg['name']}_{target_dim}.npy", reduced_queries.astype(np.float32))

            summary = summarize_experiment(
                model_name=model_cfg["name"],
                model_dim=record_embeddings.shape[1],
                target_dim=target_dim,
                records=records,
                queries=queries,
                record_embeddings=reduced_records,
                query_embeddings=reduced_queries,
                k_values=(1, 3, 5, 10),
            )
            summaries.append(summary)

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary_path = OUTPUT_DIR / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved experiment summary to {summary_path}")
    return summary_df


def plot_pca(points: np.ndarray, labels: Sequence[str], output_path: Path, title: str, category_names: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    normalized_labels = [str(label or "uncategorized") for label in category_names]
    for category in sorted(set(normalized_labels)):
        mask = np.array([label == category for label in normalized_labels])
        if not np.any(mask):
            continue
        ax.scatter(points[mask, 0], points[mask, 1], alpha=0.7, label=category)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_umap(points: np.ndarray, labels: Sequence[str], output_path: Path, title: str, category_names: Sequence[str]) -> None:
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embed_2d = reducer.fit_transform(points)
    fig, ax = plt.subplots(figsize=(10, 7))
    normalized_labels = [str(label or "uncategorized") for label in category_names]
    for category in sorted(set(normalized_labels)):
        mask = np.array([label == category for label in normalized_labels])
        if not np.any(mask):
            continue
        ax.scatter(embed_2d[mask, 0], embed_2d[mask, 1], alpha=0.75, label=category)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def generate_visualization_files() -> None:
    records = load_records()
    category_labels = [str(record.get("category") or "uncategorized") for record in records]
    embeddings = np.load(OUTPUT_DIR / "medical_embeddings_MiniLM.npy")
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(embeddings)
    plot_pca(reduced, category_labels, OUTPUT_DIR / "pca_visualization.png", "PCA projection of medical records", category_labels)
    plot_umap(embeddings, category_labels, OUTPUT_DIR / "umap_visualization.png", "UMAP projection of medical records", category_labels)
    print("Saved PCA and UMAP visualizations to the outputs directory.")


def main() -> None:
    run_experiments()
    generate_visualization_files()
    print("Semantic retrieval pipeline completed successfully.")


if __name__ == "__main__":
    main()
