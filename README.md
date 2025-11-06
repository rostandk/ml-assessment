# Keyword Spam Moderation – Colab Workflow

This repository now centres on a single Google Colab notebook that reproduces the
entire multimodal keyword spam pipeline end-to-end: we review the legacy "Old
Way", train a leakage-free TF‑IDF baseline, fine-tune `Qwen/Qwen3-VL-2B-Instruct`
with Unsloth QLoRA, run deterministic Transformers inference, sweep policy
thresholds, curate qualitative examples, and package every artifact for download.
All previous implementations (code, infra, notebooks, docs) live under
`archive/` for reference.

## Prerequisites

- Python 3.11+ and [uv](https://github.com/astral-sh/uv) for local tooling/tests.
- Optional: Hugging Face token if the chosen checkpoint is gated.
- Data files: `data/train_set.tsv` and `data/test_set.tsv` from the assignment.

## 1. Install Local Dependencies

```bash
uv sync
```

Run the unit tests to verify the helper package builds correctly:

```bash
uv run pytest
```

## 2. (Optional) Publish Downloaded Images

The Colab notebook downloads every `image_url` on demand, so no pre-built cache
is required. After a run completes, you may optionally publish the cached images
(to speed up future runs or share with teammates):

```bash
uv run python utils.py sync-gcs \
    --source /content/cache/images \
    --bucket ml-assesment \
    --prefix images \
    --public
```

Or via `gsutil`:

```bash
gsutil -m rsync -r /content/cache/images gs://ml-assesment/images
gsutil -m acl set -R -a public-read gs://ml-assesment/images
```

This step is optional—the notebook functions even when the bucket is empty.

## 3. Stage the TSV Data

Copy `data/train_set.tsv` and `data/test_set.tsv` somewhere the notebook can
reach:

- **Recommended**: upload both files to Google Drive and note their paths.
- Alternatively, upload them directly in the Colab session (Files tab → Upload).

## 4. Run the Colab Notebook

1. Open `keyword_spam.ipynb` in Google Colab (GPU runtime – A100 preferred; T4
   works with smaller batch sizes).
2. Execute the notebook from top to bottom. The first setup cells automatically
   clone this repository into the Colab runtime so that `utils.py`, `data/`, and
   other assets are available (`sys.path` is updated for you). The cells are
   organised into the following sections (key narrative extras included by
   default):
   - **Executive Introduction** – concise overview of the problem, target, and approach.
   - **Junior Notebook Review (Old Way)** – small, concrete examples showcasing weaknesses
     of the legacy method (hashtag spam, CTAs, brand mentions).
   - **What We Predict** – target, label confidence, and how thresholds map to policy.
   - **Setup** – installs pinned dependencies, mounts Drive, prints a GPU “run
     plan”, and defines helper utilities (URL normalisation, download/cache,
     schema checks).
   - **Baseline** – trains a TF‑IDF + logistic regression model fitted only on
     the training split (no leakage) and logs metrics to
     `artifacts/baseline_metrics.json`.
   - **SFT Preparation** – downloads images from their original URLs (with
     retries) into `/content/cache/images`, logs successes/failures, and builds
     Unsloth-ready JSONL datasets. If an image is missing, prompts explicitly
     note “image unavailable” so inference mirrors the text-only fallback.
   - **Fine-Tuning** – runs Unsloth QLoRA with early stopping and
     `metric_for_best_model="eval_macro_f1"`; saves the merged adapters to
     `artifacts/merged_model/`.
   - **Inference & Evaluation** – generates deterministic JSON responses,
     performs a threshold sweep for the demotion policy, records metrics, and
     writes `predictions_with_decisions.parquet` plus `metrics.json`.
   - **Curated Gallery** – displays TP/TN/FP/FN examples (with images when
     available) so you can review successes and failure modes.
   - **Packaging** – bundles everything into
     `artifacts/keyword_spam_artifacts.zip` for download. Optional cells show how
     to export the cache or artifacts back to Drive or GCS.
3. Download the ZIP (or copy it to Drive) when the final cell completes.

## 5. Supporting Utilities

- `utils.py`
  - URL normalisation, hashed filename generation, on-demand download helpers,
    and the optional `sync-gcs` CLI (run via `python utils.py sync-gcs --source ...`).
- `docs/plan_collab.md`
  - The implementation blueprint used to build the notebook.

## Legacy Materials

All prior experiments, infra templates, and vendor notes remain available under
`archive/`. Consult them if you need to reconstruct the earlier Ray/MLflow-based
pipeline or review historical design decisions.
