# Medical Record Semantic Retrieval

This project searches medical records by meaning instead of relying only on
exact keyword matches. It converts medical notes and user queries into
sentence embeddings, compares the vectors, and returns the most similar
records.

## 1. How the project works

The complete workflow is:

1. Load medical records and search queries from JSON files.
2. Clean the text by removing extra whitespace and ignoring invalid or empty
   entries.
3. Load one or more Sentence Transformer embedding models.
4. Convert every medical note and query into a numerical embedding vector.
5. Normalize the vectors so cosine similarity can be calculated efficiently.
6. Store the generated vectors as `.npy` files in the `outputs` directory.
7. Build a FAISS similarity index for fast nearest-neighbor search.
8. Retrieve the top matching records for each query.
9. Calculate retrieval results using cosine similarity, dot product, and
   Euclidean distance.
10. Reduce embedding dimensions with PCA for analysis and visualization.
11. Generate PCA and UMAP plots to visualize the record embeddings.
12. Save experiment summaries and retrieval results as CSV files.

## 2. Project structure

```text
Semantic retrieval/
├── data/
│   ├── medical_records 1.json
│   └── queries 1.json
├── outputs/
├── Requirements.txt
├── semantic_pipeline.py
├── embed.py
├── demo_query.py
├── retrieve_all_records.py
├── similarity_measures.py
├── embeddings_reduction.py
├── plotting.py
├── ploting.py
└── model_selection.py
```

### Input files

#### `data/medical_records 1.json`

This file contains the medical record collection. Each record should contain
at least:

```json
{
  "record_id": "MR0001",
  "note_text": "The patient's medical note goes here."
}
```

Additional fields, such as `category`, are preserved and can be used for
visualization.

#### `data/queries 1.json`

This file contains search queries. Each query should contain:

```json
{
  "query_id": "Q001",
  "query_text": "shortness of breath and leg swelling",
  "relevant_record_ids": ["MR0001"]
}
```

The `relevant_record_ids` field is used when calculating evaluation metrics.

## 3. Installation on Windows

Open PowerShell in the project directory:

```powershell
cd "C:\Semantic retrieval"
```

### Create the virtual environment

The project should be run in the local `.venv` environment. This avoids
conflicts with other Python installations and prevents the broken global Torch
installation issue that can occur with Windows Store Python.

```powershell
python -m venv .venv
```

### Activate the environment

```powershell
.venv\Scripts\activate
```

After activation, PowerShell should show `(.venv)` at the beginning of the
command prompt.

### Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r Requirements.txt
```

The dependencies include:

- `sentence-transformers` for generating text embeddings.
- `faiss-cpu` for fast vector similarity search.
- `numpy` for numerical arrays and saved embeddings.
- `scikit-learn` for PCA and similarity calculations.
- `pandas` for tabular retrieval results and experiment summaries.
- `matplotlib` for PCA plots.
- `umap-learn` for UMAP visualization.
- `joblib` for saving PCA models.

## 4. Run the complete pipeline

Run the main pipeline with:

```powershell
python semantic_pipeline.py
```

This command:

1. Loads the medical records and queries.
2. Runs all configured embedding models.
3. Saves embeddings for MiniLM, MPNet, and MultiQA.
4. Creates PCA-reduced versions of the embeddings.
5. Evaluates retrieval performance.
6. Saves `experiment_summary.csv`.
7. Creates PCA and UMAP visualizations.

The first run downloads the configured models from Hugging Face. Internet
access is therefore required the first time each model is used. Later runs
reuse the local model cache.

## 5. Generate embeddings only

To generate record and query embeddings without running the complete
visualization pipeline:

```powershell
python embed.py
```

The configured models are:

| Name | Model | Purpose |
|---|---|---|
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` | Lightweight general-purpose baseline |
| MPNet | `sentence-transformers/all-mpnet-base-v2` | Higher-quality semantic embeddings |
| MultiQA | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Retrieval-focused embeddings |

The MiniLM embeddings are also copied to these compatibility files:

```text
outputs/medical_embeddings.npy
outputs/query_embeddings.npy
```

## 6. Search with a custom query

Run the interactive command-line search script:

```powershell
python demo_query.py
```

The default query is:

```text
leg swelling and shortness of breath
```

Provide a different query and result count:

```powershell
python demo_query.py "persistent cough with fever" --k 10
```

The script embeds the query, creates a cosine-similarity FAISS index from the
medical record embeddings, and prints the highest-ranked record IDs and
similarity scores.

## 7. Retrieve all records for all stored queries

Run:

```powershell
python retrieve_all_records.py
```

For every query, this script calculates the top five records using:

- Cosine similarity.
- Dot product.
- Euclidean distance.

The combined results are saved to:

```text
outputs/retrieval_results_all_queries.csv
```

## 8. Compare similarity measures

Run:

```powershell
python similarity_measures.py
```

This uses the first stored query and prints the top records for cosine
similarity, dot product, and Euclidean distance.

## 9. Reduce embedding dimensions

Run:

```powershell
python embeddings_reduction.py
```

This applies PCA to the default medical embeddings and saves:

```text
outputs/medical_embeddings_reduced.npy
outputs/pca_model.joblib
```

The default target dimension is 128, or the original dimension if the
embedding size is already smaller than 128.

## 10. Create visualizations

For a basic PCA scatter plot, run either:

```powershell
python plotting.py
```

or:

```powershell
python ploting.py
```

Both scripts save:

```text
outputs/pca_visualization.png
```

The complete `semantic_pipeline.py` additionally creates a UMAP plot:

```text
outputs/umap_visualization.png
```

The UMAP plot groups points using the record `category` field when available.

## 11. Compare embedding models

Run:

```powershell
python model_selection.py
```

This loads each configured model, embeds a small set of sample medical
sentences, and reports:

- Embedding dimensions.
- Encoding time.
- Model name and Hugging Face model ID.

## 12. Important generated files

The `outputs` directory may contain:

| File | Description |
|---|---|
| `medical_embeddings.npy` | Default MiniLM record embeddings |
| `query_embeddings.npy` | Default MiniLM query embeddings |
| `medical_embeddings_<model>.npy` | Record embeddings for a specific model |
| `query_embeddings_<model>.npy` | Query embeddings for a specific model |
| `medical_embeddings_reduced.npy` | PCA-reduced record embeddings |
| `pca_model.joblib` | Saved PCA transformer |
| `retrieval_results_all_queries.csv` | Retrieval results for all queries |
| `experiment_summary.csv` | Precision, recall, and MRR experiment results |
| `pca_visualization.png` | PCA scatter plot |
| `umap_visualization.png` | UMAP scatter plot |

## 13. Evaluation metrics

The pipeline evaluates retrieval using:

- **Precision@K**: the proportion of the top K results that are relevant.
- **Recall@K**: the proportion of all relevant records found in the top K.
- **MRR**: the reciprocal rank of the first relevant result.

These metrics require each query to have correct
`relevant_record_ids` values in `data/queries 1.json`.

## 14. Recommended execution order

For a fresh checkout, use:

```powershell
cd "C:\Semantic retrieval"
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r Requirements.txt
python semantic_pipeline.py
python demo_query.py "leg swelling and shortness of breath" --k 5
python retrieve_all_records.py
```

## 15. Troubleshooting

### `ModuleNotFoundError: No module named 'umap'`

Make sure the virtual environment is activated and install the requirements:

```powershell
.venv\Scripts\activate
python -m pip install -r Requirements.txt
```

The import name is `umap`, but the package name in `Requirements.txt` is
`umap-learn`.

### Python uses the wrong environment

Check the Python executable:

```powershell
python -c "import sys; print(sys.executable)"
```

It should point to:

```text
C:\Semantic retrieval\.venv\Scripts\python.exe
```

If it does not, activate the environment again:

```powershell
.venv\Scripts\activate
```

### Models cannot be downloaded

The first execution requires internet access to download the Sentence
Transformer models. Check the connection and run the command again. Once
downloaded, the models are loaded from the local Hugging Face cache.

### Missing output files

Run `embed.py` or `semantic_pipeline.py` first. Scripts such as
`demo_query.py`, `similarity_measures.py`, and `retrieve_all_records.py`
expect the embedding files to already exist in `outputs`.
