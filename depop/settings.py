from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

try:  # pragma: no cover - torch may be absent in some environments
    import torch
except ModuleNotFoundError:  # type: ignore
    torch = None  # type: ignore

try:
    import config  # user-facing constants
except Exception:  # pragma: no cover
    config = None


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def _linspace(start: float, stop: float, num: int) -> tuple[float, ...]:
    if num == 1:
        return (round(start, 2),)
    step = (stop - start) / (num - 1)
    return tuple(round(start + i * step, 2) for i in range(num))


@dataclass
class PathConfig:
    repo_dir: Path
    data_train: Path
    data_test: Path
    cache_dir: Path
    artifacts_dir: Path
    cache_zip_path: Path
    publish_cache_zip_path: Path
    sft_dir: Path
    model_dir: Path
    trainer_dir: Path
    predictions_path: Path
    metrics_path: Path
    classification_report_path: Path
    packaged_zip: Path


@dataclass
class GCSConfig:
    enabled: bool
    bucket: str
    images_prefix: str = "images"
    datasets_prefix: str = "datasets"
    models_prefix: str = "models"
    artifacts_prefix: str = "artifacts"


@dataclass
class TrainingConfig:
    model_id: str
    dtype: str
    epochs: int
    learning_rate: float
    max_seq_len: int
    warmup_ratio: float
    batch_size_t4: int
    grad_accum_t4: int
    batch_size_a100: int
    grad_accum_a100: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    prompt_template: str


@dataclass
class PolicyConfig:
    review_threshold: float
    demote_threshold: float
    review_grid: Sequence[float]
    demote_grid: Sequence[float]


@dataclass
class NotebookSettings:
    repo_url: str
    is_colab: bool
    seed: int
    paths: PathConfig
    gcs: GCSConfig
    training: TrainingConfig
    policy: PolicyConfig

    def summary(self) -> str:
        payload = {
            "repo": str(self.paths.repo_dir),
            "artifacts": str(self.paths.artifacts_dir),
            "cache": str(self.paths.cache_dir),
            "model": self.training.model_id,
        }
        return json.dumps(payload, indent=2)


def _default_dtype() -> str:
    if torch is None:  # pragma: no cover - when torch is unavailable
        return "float16"
    try:
        return "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    except Exception:  # pragma: no cover - torch runtime issues
        return "float16"


def load_settings(overrides: Optional[dict[str, Any]] = None) -> NotebookSettings:
    overrides = overrides or {}
    is_colab = overrides.get("is_colab") or ("google.colab" in globals().get("sys", __import__("sys")).modules)
    repo_url = overrides.get("repo_url", "https://github.com/rostandk/ml-assessment.git")
    repo_dir = Path(overrides.get("repo_dir", "/content/ml-assessment" if is_colab else Path.cwd()))
    data_dir = Path(overrides.get("data_dir", repo_dir / "data"))
    cache_dir = Path(overrides.get("cache_dir", "/content/cache/images" if is_colab else repo_dir / "cache" / "images"))
    artifacts_dir = Path(overrides.get("artifacts_dir", "/content/artifacts" if is_colab else repo_dir / "artifacts"))

    # Paths prefer config.py values when available
    paths = PathConfig(
        repo_dir=repo_dir,
        data_train=Path(overrides.get("train_tsv", data_dir / "train_set.tsv")),
        data_test=Path(overrides.get("test_tsv", data_dir / "test_set.tsv")),
        cache_dir=Path(getattr(config, "LOCAL_IMAGE_CACHE", str(cache_dir))) if config else cache_dir,
        artifacts_dir=Path(getattr(config, "ARTIFACTS_DIR", str(artifacts_dir))) if config else artifacts_dir,
        cache_zip_path=Path(cache_dir.parent / "images_cache.zip"),  # unused with new plan
        publish_cache_zip_path=Path(cache_dir.parent / "exported_images_cache.zip"),  # unused
        sft_dir=Path(getattr(config, "SFT_DIR", str(artifacts_dir / "sft"))) if config else artifacts_dir / "sft",
        model_dir=Path(getattr(config, "MODEL_DIR", str(artifacts_dir / "merged_model"))) if config else artifacts_dir / "merged_model",
        trainer_dir=Path(getattr(config, "TRAINER_DIR", str(artifacts_dir / "trainer"))) if config else artifacts_dir / "trainer",
        predictions_path=artifacts_dir / "predictions_with_decisions.parquet",
        metrics_path=artifacts_dir / "metrics.json",
        classification_report_path=artifacts_dir / "classification_report.json",
        packaged_zip=artifacts_dir / "keyword_spam_artifacts.zip",
    )

    gcs = GCSConfig(
        enabled=bool(overrides.get("imagestore_enabled", getattr(config, "GCS_ENABLED", True) if config else True)),
        bucket=str(overrides.get("gcs_bucket", getattr(config, "GCS_BUCKET", "ml-assesment") if config else "ml-assesment")),
        images_prefix=str(overrides.get("gcs_images_prefix", getattr(config, "GCS_IMAGES_PREFIX", "images") if config else "images")),
        datasets_prefix=str(overrides.get("gcs_datasets_prefix", getattr(config, "GCS_DATASETS_PREFIX", "datasets") if config else "datasets")),
        models_prefix=str(overrides.get("gcs_models_prefix", getattr(config, "GCS_MODELS_PREFIX", "models") if config else "models")),
        artifacts_prefix=str(overrides.get("gcs_artifacts_prefix", getattr(config, "GCS_ARTIFACTS_PREFIX", "artifacts") if config else "artifacts")),
    )

    training = TrainingConfig(
        model_id=overrides.get("model_id", getattr(config, "MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct") if config else "Qwen/Qwen3-VL-2B-Instruct"),
        dtype=overrides.get("dtype", getattr(config, "DTYPE", _default_dtype()) if config else _default_dtype()),
        epochs=int(overrides.get("epochs", 3)),
        learning_rate=float(overrides.get("learning_rate", getattr(config, "LEARNING_RATE", 1e-4) if config else 1e-4)),
        max_seq_len=int(overrides.get("max_seq_len", getattr(config, "MAX_SEQ_LEN", 1024) if config else 1024)),
        warmup_ratio=float(overrides.get("warmup_ratio", getattr(config, "WARMUP_RATIO", 0.05) if config else 0.05)),
        batch_size_t4=int(overrides.get("batch_size_t4", getattr(config, "BATCH_SIZE_T4", 4) if config else 4)),
        grad_accum_t4=int(overrides.get("grad_accum_t4", getattr(config, "GRAD_ACCUM_T4", 4) if config else 4)),
        batch_size_a100=int(overrides.get("batch_size_a100", getattr(config, "BATCH_SIZE_A100", 8) if config else 8)),
        grad_accum_a100=int(overrides.get("grad_accum_a100", getattr(config, "GRAD_ACCUM_A100", 2) if config else 2)),
        lora_r=int(overrides.get("lora_r", getattr(config, "LORA_R", 16) if config else 16)),
        lora_alpha=int(overrides.get("lora_alpha", getattr(config, "LORA_ALPHA", 32) if config else 32)),
        lora_dropout=float(overrides.get("lora_dropout", getattr(config, "LORA_DROPOUT", 0.05) if config else 0.05)),
        prompt_template=overrides.get(
            "prompt_template",
            "You are a moderator. Respond with JSON containing is_spam (bool), confidence (0-1), reason (short).",
        ),
    )

    policy = PolicyConfig(
        review_threshold=float(overrides.get("review_threshold", getattr(config, "REVIEW_THRESHOLD", 0.5) if config else 0.5)),
        demote_threshold=float(overrides.get("demote_threshold", getattr(config, "DEMOTE_THRESHOLD", 0.7) if config else 0.7)),
        review_grid=tuple(overrides.get("review_grid", _linspace(0.1, 0.9, 9))),
        demote_grid=tuple(overrides.get("demote_grid", _linspace(0.2, 1.0, 9))),
    )

    return NotebookSettings(
        repo_url=repo_url,
        is_colab=bool(is_colab),
        seed=int(overrides.get("seed", 42)),
        paths=paths,
        gcs=gcs,
        training=training,
        policy=policy,
    )
