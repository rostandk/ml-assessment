# Keyword Spam Moderation

Single-notebook workflow (Google Colab friendly) powered by a modular Python package (`depop`) that handles data loading, caching, baseline benchmarking, Unsloth QLoRA fine-tuning, inference, policy evaluation, and artifact packaging. The notebook is narrative-first: every heavy step is delegated to the workflow API, so cells stay visual and easy to reason about.

## Quick Start

### Local setup

```bash
uv sync              # installs base deps + dev extras
uv run pytest -q     # run unit tests
```

### Colab setup

At the top of `keyword_spam.ipynb`, run the install cell:

```python
%pip install '.[colab]'
```

Upload `data/train_set.tsv` and `data/test_set.tsv` (Drive or Colab Files tab), then execute the notebook top-to-bottom. The first cells:

1. Install the extras listed in `pyproject.toml` (`.[colab]`).
2. Clone this repo into `/content/ml-assessment`.
3. Initialize the workflow helpers (`RepoManager`, `ImageStore`, `DataModule`, `BaselineModel`, `SFTDatasetBuilder`, `QwenTrainer`, `InferenceRunner`, `EvaluationSuite`, `ArtifactManager`).

A generated Table of Contents (TOC) lets you jump between sections if you need to revisit a stage.

## Architecture Overview

```
Data TSVs ──► DataModule.load() ──► ImageStore.ensure_all()
    │                              │
    │                              └─► download + public GCS upload + manifest
    │
    ├─► BaselineModel.run() ──► baseline_metrics.json
    │
    └─► SFTDatasetBuilder.build() ──► train/val JSONL
                                      │
                                      └─► QwenTrainer.train() ──► merged_model/
                                                            │
                                                            └─► InferenceRunner.predict()
                                                                      │
                                                                      ├─► EvaluationSuite.threshold_sweep/evaluate
                                                                      ├─► EvaluationSuite.build_gallery/show_legacy_failures
                                                                      └─► ArtifactManager.save_* / package
```

**Key modules (strict imports at top)**

- `depop.settings`: dataclasses + `load_settings()`/`setup_logging()`; detects Colab/local paths, handles dtype defaults.
- `depop.repo`: clone/pull + sys.path injection.
- `depop.media`: `ImageStore` for deterministic path‑based naming, download, and public GCS upload with idempotency + manifest.
- `depop.data`: TSV loader (schema validation + `label_confidence`), `DataModule`, `BaselineModel`, `SFTDatasetBuilder`.
- `depop.training`: `QwenTrainer` (Unsloth + TRL). Imports torch/unsloth/trl lazily and raises a clear error if they’re missing.
- `depop.inference`: `InferenceRunner` with deterministic decoding and optional micro-batching for text-only rows; handles image availability flags consistently.
- `depop.evaluation`: `EvaluationSuite` for threshold sweep, policy metrics, curated gallery, and a reusable “Old Way” failure sampler.
- `depop.artifacts`: write metrics/predictions/classification reports and package artifacts.

`utils.py` now re-exports the essentials for backwards compatibility, but the notebook imports from the `depop` package directly.

## Notebook Flow (TOC anchors)

1. **Executive Introduction** (`#exec-intro`)
2. **Environment Setup** (`#env-setup`) – installs `.[colab]`, clones repo, prints GPU run plan, seeds RNGs.
3. **Old Way Review** (`#old-way`) – `EvaluationSuite.show_legacy_failures(train_df)` renders the historic weaknesses (hashtags, CTAs, brand mentions).
4. **Data Preparation / What We Predict** (`#data-prep`) – TSV load, schema validation, `label_confidence`, class balance plot.
5. **Baseline** (`#baseline`) – TF‑IDF + logistic regression logged to `baseline_metrics.json` via `BaselineModel`.
6. **SFT Dataset** (`#sft`) – `SFTDatasetBuilder` mirrors prompts for text-only fallback (“image unavailable” note).
7. **Fine-tuning** (`#train`) – `QwenTrainer` (Unsloth QLoRA + SFTTrainer, early stopping, auto batch/grad accumulation).
8. **Inference** (`#infer`) – `InferenceRunner` (deterministic decoding; optional micro-batching for text-only rows).
9. **Evaluation** (`#eval`) – `EvaluationSuite.threshold_sweep/evaluate`, metrics stored in `metrics.json` + `classification_report.json`.
10. **Curated Gallery** (`#gallery`) – HTML grid for TP/TN/FP/FN (skips missing source rows gracefully).
11. **Artifacts** (`#artifacts`) – `ArtifactManager` writes predictions, metrics, manifest CSVs, and a packaged zip for hand-offs.

## Image Cache & Public GCS

- `ImageStore.ensure_all()` deterministically maps each URL to `cache/images/<host>/<path>` and downloads only the missing files.
- Every successful download is uploaded to your configured public GCS bucket so that future runs can skip network hits. Configure the bucket/prefixes in `config.py`.
- `ArtifactManager.save_manifest()` writes `artifacts/image_manifest.csv`, listing `image_url`, `relative_path`, `local_path`, and the resulting public URL. Share this CSV with collaborators or feed it into offline batch jobs.
- Grant anonymous read access once:

```bash
gsutil iam ch allUsers:objectViewer gs://ml-assesment
```

- Because filenames are derived from URL paths, repeated runs will reuse and re-upload the same objects idempotently.

## Configuration Reference (selected keys)

| Key | Description |
| --- | --- |
| `model_id` | Hugging Face checkpoint (default `Qwen/Qwen3-VL-2B-Instruct`). |
| `dtype` | Auto-detected (`bfloat16` when supported, else `float16`). Override if needed. |
| `GCS_BUCKET` / `GCS_IMAGES_PREFIX` | Public bucket + prefix used by `ImageStore.upload()` and manifest publication. |
| `LOCAL_IMAGE_CACHE` | Folder where images land on Colab (`/content/cache/images` by default). |
| `review_threshold` / `demote_threshold` | Default policy thresholds (grid sweep refines them). |

All paths (repo/data/cache/artifacts/sft/model/trainer) are defined in `PathConfig` and respect Colab vs local environments.

## Testing

- `uv run --extra dev pytest -q`
  - `tests/test_data.py`: TSV loader validation + schema guards.
  - `tests/test_workflow.py`: `ImageStore` concurrency, `ArtifactManager` uploads/manifest, trainer wiring (mocked), inference runner.

GPU- and network-heavy components are mocked, so tests run quickly on dev machines.

## Legacy Artifacts

Everything prior to the Colab-only refactor (Ray pipelines, notebooks, docs) lives under `archive/`. Keep them for historical reference or migration notes, but the supported workflow is entirely handled by `depop` + `keyword_spam.ipynb`.
