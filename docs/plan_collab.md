# Colab-Only Plan (Verified, Detailed, End-to-End)

One Colab notebook handles the full workflow: baseline (TF‑IDF), Unsloth QLoRA fine‑tuning on Qwen3‑VL, Transformers inference, evaluation with threshold sweep, curated qualitative analysis, uplift vs baseline, and artifact packaging. No servers and no vLLM. Plan integrates critical fixes (image cache mapping, correct multimodal message format, early stopping with compute_metrics) and minor refinements for clarity, reproducibility, and analytical depth.

## Assumptions & Environment
- Colab Pro/Pro+ available; prefer A100; T4 works with 4‑bit QLoRA.
- Model for training: `Qwen/Qwen3-VL-2B-Instruct` (fits T4 + QLoRA). Larger variants may be used for inference‑only on A100.
- Data TSVs have columns: `product_id`, `description`, `image_url`, `label` (0/1), `yes_count`, `no_count`.
- No pre-populated image cache required; the notebook downloads each `image_url`
  on demand and can optionally export the cache afterwards. If an image is
  missing, prompts explicitly include a note ("image unavailable") so the model
  bases its judgment on text only; inference mirrors this behavior.
- All inference/evaluation uses Transformers (no vLLM).

## Data & Assets (Download on Demand)
- The notebook no longer assumes a pre-populated `cache/images` directory or a
  pre-synced GCS bucket. Instead, it downloads each `image_url` directly from
  the source site when needed, stores it under `/content/cache/images`, and logs
  whether the download succeeded. Missing images simply fall back to text-only
  prompts; evaluation slices report “image present vs missing” so the impact is
  visible.
- The new helpers `utils.download_image` / `utils.download_images` mirror the
  legacy hashing scheme, retry with exponential backoff, validate content type,
  and run in parallel for speed. They return ``None`` after exhausting their
  attempts, and the notebook calls these utilities directly so the tested code
  path is the one exercised in Colab.
- After the download step, two **optional** export paths are provided inside the
  notebook:
  1. **Save to Drive** – zip or copy `/content/cache/images` into Google Drive
     for later reuse.
  2. **Publish to GCS** – run `python utils.py sync-gcs --source
     /content/cache/images --bucket ml-assesment --prefix images --public` if you
     want a reusable, publicly hosted cache. This is helpful but *not* required
     for the workflow to function.
- Because the cache is rebuilt lazily, the repository can stay lightweight when
  pushed to GitHub; users simply run the notebook to regenerate any missing
  assets.

## Runtime Setup (Colab)
- Mount Drive at `/content/drive`; set `DRIVE_ROOT=/content/drive/MyDrive/keyword_spam_vlm`.
- Create: `artifacts/`, `datasets/`, `cache/images/`.
- Install pinned deps to avoid Colab drift:
  - `pip install -U "transformers==4.44.*" "accelerate==0.34.*" "peft==0.12.*" "datasets==2.20.*" unsloth bitsandbytes pillow pandas scikit-learn pyarrow tqdm`
- Print CUDA/torch/bitsandbytes; set RNG seeds (`random`, `numpy`, `torch`).
- Show a “Run Plan” banner: detected GPU (A100/T4), recommended `batch_size`/`grad_accum`, sanity/full run guidance.

## Configuration & Auto‑Tuning
- Config dict at top:
  - `model_id='Qwen/Qwen3-VL-2B-Instruct'`
  - `dtype='bfloat16' if torch.cuda.is_bf16_supported() else 'float16'`
  - QLoRA: `load_in_4bit=True`, `lora_r=16`, `lora_alpha=32`, `lora_dropout=0.05`
  - Train: `epochs=3`, `lr=1e-4`, `max_seq_len=1024`, `warmup_ratio=0.05`, `save_each_epoch=True`
  - Batch/accum (auto): T4 → `batch_size=4`, `grad_accum=4`, gradient checkpointing ON; A100 → `batch_size=8–16`, `grad_accum=2–4`
  - Paths: `DATA_TRAIN`, `DATA_TEST`, `IMAGE_CACHE_DIR`, `RUN_DIR`, `MERGED_DIR`
  - Policy thresholds: `review_threshold=0.5`, `demote_threshold=0.7`
  - `seed=42`, `max_rows=None` (set small for sanity)
- Persist `run_metadata.json` at end (GPU, sizes, timings, thresholds, metrics).

## Data Loading & Validation
- Load TSVs (`sep='\t'`), normalize column names; drop unlabeled rows for SFT targets.
- Validate schema: `product_id` non‑empty & unique; `label` ∈ {0,1}; `yes_count`,`no_count` ≥ 0.
- Compute `label_confidence=(yes-no)/(yes+no)` with zero‑safe handling; summarize.
- Stratified 90/10 train/val split from labeled rows; print sizes and class balance.
- Optional label quality handling (toggle): weight by `label_confidence` or exclude ambiguous labels; default is include all. If enabled, report both results.

## Old Way (Junior Notebook Review)
- Markdown recap: text‑only heuristics, brittle features, no reproducibility.
- Mini table of 3–5 failure cases (brand mismatch, hashtag spam, CTA/“DM for discounts”); will be revisited later.

## Baseline: TF‑IDF + Logistic Regression (No Leakage)
- Fit TF‑IDF only on training descriptions:
  - `vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')`
  - `X_train = vectorizer.fit_transform(train_df['description'])`
  - `X_val   = vectorizer.transform(val_df['description'])`
- Train `LogisticRegression` (optionally `class_weight='balanced'`), evaluate on validation only.
- Metrics: macro/micro F1, precision, recall, accuracy; confusion matrix; optional AUPRC (spam).
- Save `baseline_metrics.json`; show concise summary.

## Image Cache Integration (Critical Fix: URL→Local Path)
- Implement `get_image_path_from_url(url) -> str` in notebook:
  - Compute a deterministic filename (e.g., `sha1(url).hexdigest()`), preserve extension if available (fallback `.jpg`).
  - Return `os.path.join(IMAGE_CACHE_DIR, filename)`.
- Always pass local file paths to the model. If not found, proceed text‑only. Track counts and list missing items.

## SFT Dataset Build (Correct Multimodal Message Format)
- System prompt: enforce strict JSON only with fields `is_spam` (bool), `confidence` (0–1), `reason` (short).
- Build Unsloth `messages` per example with correct content ordering for Qwen3‑VL:
  - If image exists:
    - `user_content = [{"type":"image","image": image_path}, {"type":"text","text": f"{PROMPT_TEMPLATE}\n\nDescription: {desc}"}]`
  - Else (text only):
    - `user_content = [{"type":"text","text": f"{PROMPT_TEMPLATE}\n\nDescription: {desc}"}]`
- Assistant target for labeled rows: minified JSON string derived from ground truth (confidence=1.0).
- Save `train.jsonl`, `val.jsonl`; log counts (with/without images) & preview a few rows.

## Training (Unsloth QLoRA) with Early Stopping
- Load datasets via `datasets.load_dataset('json', ...)`; map to `messages` if missing.
- `FastVisionModel.from_pretrained(model_id, dtype, max_seq_length, load_in_4bit=True)`; then `get_peft_model(r, alpha, dropout)`.
- `VisionSFTTrainer` + `TrainingArguments`:
  - epochs, per‑device batch size, gradient accumulation, lr, warmup; `bf16=True` when supported;
  - `evaluation_strategy="epoch"`, `save_strategy="epoch"`;
  - `load_best_model_at_end=True`, `metric_for_best_model="eval_macro_f1"`, `greater_is_better=True`;
  - `predict_with_generate=True`, `generation_max_length=256` (to enable text generation during eval for metrics).
- `compute_metrics` (clarified):
  - With `predict_with_generate=True`, `eval_preds.predictions` (or first element) are generated sequences. Decode to strings, parse JSON strictly to `is_spam` labels; compare against ground truth; compute macro/micro F1, precision, recall, accuracy; return a dict including `macro_f1` (name matches `metric_for_best_model`).
  - If the trainer backend does not return decoded text on this model, fall back to a validation hook that runs generation on the val set post‑epoch, computes the same metrics, and updates a tracked `best_score` used by early stopping.
- Early stopping:
  - `from transformers import EarlyStoppingCallback`
  - `callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]`
- Train; save checkpoints each epoch (optional). Keep and merge the best model; save merged `MERGED_DIR` (weights, tokenizer, image_processor).

## Inference with Transformers (Deterministic for Evaluation)
- Recreate messages using the same system prompt & user content; attach local image path if present.
- Decoding defaults for evaluation: `temperature=0.0`, `top_p=1.0`, `max_new_tokens=256` (use `temperature=0.2` for qualitative demos if desired).
- Micro‑batching to avoid OOM; bound total tokens by `max_seq_len`.
- JSON parsing & validation: robust extraction (first JSON block), strict schema validation (`pydantic`/`jsonschema`), record malformed %; keep a `sample_responses.jsonl` for inspection.
- Save predictions: `validation_predictions.parquet` (+ CSV), optional `test_predictions.parquet`.

## Evaluation, Policy, & Threshold Sweep
- Metrics (validation and optional test): macro/micro F1, precision, recall, accuracy; confusion matrix; optional AUPRC.
- Demotion policy:
  - demote if `is_spam and confidence >= demote_threshold`
  - review if `is_spam and review_threshold <= confidence < demote_threshold`
  - keep otherwise
- Threshold sweep:
  - Grid over `(review_threshold, demote_threshold)` to maximize macro F1 (or cost‑weighted utility). Apply best thresholds, recompute decisions/metrics.
- Slice metrics: report “image available vs image missing” subsets to quantify image contribution.
- Save: `metrics.json`, `classification_report.json`, `predictions_with_decisions.parquet` (+ CSV if useful).

## Curated Qualitative Gallery (Insightful, Not Random)
- Build a targeted gallery across error modes:
  - True Positives: 3–4 examples (spam correctly identified)
  - True Negatives: 3–4 examples (non‑spam correctly identified)
  - False Positives (critical): 3–4 examples (non‑spam incorrectly flagged as spam)
  - False Negatives (missed spam): 3–4 examples (spam missed)
- For each: display image (if cached), description, true label, predicted label, confidence, reason, and final decision.
- Revisit the Old Way failure cases within this gallery to demonstrate concrete improvements.

## Artifacts & Packaging
- Write heavy IO to `/content` during runs; at the end, move/sync to Drive.
- Package ZIP with:
  - `merged_model/` (weights, tokenizer, image_processor)
  - `train.jsonl`, `val.jsonl`
  - `validation_predictions.parquet` (+ CSV), optional `test_predictions.parquet`
  - `metrics.json`, `classification_report.json`, `predictions_with_decisions.parquet`
  - `run_metadata.json` (config, GPU, timings, thresholds, metrics)
- Save under `DRIVE_ROOT/artifacts/` and render a Drive download link.

## Metrics & Evaluation Enhancements
- Always report macro/micro F1, precision, recall, accuracy; confusion matrix; optional AUPRC.
- Include threshold sweep plot/table and clearly indicate the chosen operating point.
- Optional cost framing: show a simple cost matrix and compute expected cost under chosen thresholds.
- Optional confidence calibration: report a small reliability summary (ECE) and plot; consider isotonic calibration on validation and evaluate impact (if time allows).

## Notebook Organization (Clarity & Reuse)
- Group logic into helper functions per section to keep cells concise and story‑like:
  - `sync_or_unzip_images()`, `get_image_path_from_url(url)`, `load_and_validate_tsvs()`
  - `run_baseline(train_df, val_df) -> metrics`
  - `build_sft_jsonl(train_df, val_df) -> paths`
  - `run_finetuning(train_jsonl, val_jsonl) -> merged_model_path`
  - `run_inference(df, model_path) -> predictions_df`
  - `evaluate(predictions_df, thresholds) -> metrics`
  - `threshold_sweep(predictions_df) -> best_thresholds`
  - `make_gallery(predictions_df, df) -> display`
- The main flow calls these functions, keeping the notebook narrative clean.

## Risks & Mitigations
- OOM on T4: smaller `batch_size`, larger `grad_accum`, gradient checkpointing, cap `max_new_tokens`; start with a sanity `max_rows` run; prefer A100 for full runs.
- Flaky images: retries/backoff; proceed text‑only; slice metrics by image availability; list missing items.
- JSON malformation: robust extraction + strict validation; log malformed ratio; store raw outputs sample.
- Session timeouts: persist intermediate outputs to Drive; provide sanity/full run switches.

## Actionable Next Steps (Implementation Order)
1) Setup cells: Drive mount, dirs, pinned deps, seeds, GPU banner.
2) Ensure `IMAGE_CACHE_DIR` via unzip/rclone/PyDrive; count present/missing images.
3) Implement `get_image_path_from_url(url)` (sha1 + extension) and verify local paths.
4) Load TSVs; schema checks; compute `label_confidence`; stratified split.
5) Baseline without leakage (fit TF‑IDF on train; transform val/test); save `baseline_metrics.json`.
6) Build SFT `messages` (image‑first content ordering) and write `train.jsonl`, `val.jsonl`.
7) Train with Unsloth QLoRA: `predict_with_generate=True`, `compute_metrics` (decode strings → parse JSON → labels), `EarlyStoppingCallback(patience=3)`, `load_best_model_at_end=True`; merge adapters.
8) Transformers inference (deterministic), micro‑batching, strict JSON parsing; save predictions.
9) Evaluate, apply demotion policy, run threshold sweep, recompute with best thresholds; slice metrics by image availability; save metrics and reports.
10) Curated qualitative gallery (TP/TN/FP/FN), revisit legacy failures.
11) Save `run_metadata.json`, package ZIP, and link for download.
