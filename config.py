"""Project configuration for Colab workflow.

Edit the constants below to change model, bucket, and paths.
This file is the single source of truth for user‑tunable settings.
"""

# -----------------------------
# Google Cloud Storage (public)
# -----------------------------
GCS_ENABLED = True
GCS_BUCKET = "ml-assesment"  # change to your bucket
GCS_IMAGES_PREFIX = "images"
GCS_DATASETS_PREFIX = "datasets"
GCS_MODELS_PREFIX = "models"
GCS_ARTIFACTS_PREFIX = "artifacts"

# -----------------------------
# Model / Training
# -----------------------------
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"  # swap easily for another checkpoint
DTYPE = "bfloat16"  # auto fallback to float16 if bf16 unsupported
EPOCHS = 3
LEARNING_RATE = 1e-4
MAX_SEQ_LEN = 1024
WARMUP_RATIO = 0.05
# Batch and accumulation (picked by GPU type in code)
BATCH_SIZE_T4 = 4
GRAD_ACCUM_T4 = 4
BATCH_SIZE_A100 = 8
GRAD_ACCUM_A100 = 2
# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# -----------------------------
# Local Paths (Colab)
# -----------------------------
LOCAL_IMAGE_CACHE = "/content/cache/images"
ARTIFACTS_DIR = "/content/artifacts"
SFT_DIR = f"{ARTIFACTS_DIR}/sft"
MODEL_DIR = f"{ARTIFACTS_DIR}/merged_model"
TRAINER_DIR = f"{ARTIFACTS_DIR}/trainer"

# -----------------------------
# Policy thresholds
# -----------------------------
REVIEW_THRESHOLD = 0.5
DEMOTE_THRESHOLD = 0.7

