from __future__ import annotations

import json
import mimetypes
import shutil
from pathlib import Path
import pandas as pd

from .settings import NotebookSettings


class ArtifactManager:
    """Persist predictions, metrics, and packaged artifacts, mirrored to GCS."""

    def __init__(self, settings: NotebookSettings):
        self.settings = settings

    # ---------------- Local saves ----------------
    def save_predictions(self, df: pd.DataFrame) -> None:
        path = self.settings.paths.predictions_path
        df.to_parquet(path, index=False)
        self._upload_file(path, self.settings.gcs.artifacts_prefix)

    def save_metrics(self, metrics: dict[str, object]) -> None:
        path = self.settings.paths.metrics_path
        path.write_text(json.dumps(metrics, indent=2))
        self._upload_file(path, self.settings.gcs.artifacts_prefix)

    def save_classification_report(self, report: str) -> None:
        path = self.settings.paths.classification_report_path
        path.write_text(report)
        self._upload_file(path, self.settings.gcs.artifacts_prefix)

    def save_manifest(self, manifest: pd.DataFrame, filename: str = "image_manifest.csv") -> Path:
        path = self.settings.paths.artifacts_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(path, index=False)
        self._upload_file(path, self.settings.gcs.artifacts_prefix)
        return path

    def package(self) -> Path:
        output = self.settings.paths.packaged_zip
        if output.exists():
            output.unlink()
        shutil.make_archive(str(output.with_suffix("")), "zip", self.settings.paths.artifacts_dir)
        self._upload_file(output, self.settings.gcs.artifacts_prefix)
        return output

    # ---------------- Dataset / model uploads ----------------
    def upload_sft_jsonl(self, train_jsonl: Path, val_jsonl: Path) -> None:
        self._upload_file(train_jsonl, self.settings.gcs.datasets_prefix)
        self._upload_file(val_jsonl, self.settings.gcs.datasets_prefix)

    def upload_model_dir(self, model_dir: Path) -> None:
        self.sync_dir(model_dir, self.settings.gcs.models_prefix)

    # ---------------- Directory sync ----------------
    def sync_dir(self, dir_path: Path, prefix: str) -> None:
        for p in Path(dir_path).rglob("*"):
            if p.is_file():
                self._upload_file(p, prefix)

    # ---------------- Helpers ----------------
    def _upload_file(self, local_path: Path, prefix: str) -> None:
        if not self.settings.gcs.enabled:
            return
        try:
            from google.cloud import storage  # pragma: no cover
        except Exception:
            return
        client = storage.Client()
        bucket = client.bucket(self.settings.gcs.bucket)
        rel_name = local_path.name
        blob_name = f"{prefix.strip('/')}/{rel_name}"
        blob = bucket.blob(blob_name)
        # idempotency: skip when sizes match
        try:
            if blob.exists() and blob.size == local_path.stat().st_size:  # type: ignore[attr-defined]
                return
        except Exception:
            pass
        ctype, _ = mimetypes.guess_type(str(local_path))
        if ctype:
            blob.content_type = ctype
        blob.cache_control = "public, max-age=31536000"
        blob.upload_from_filename(str(local_path))
        blob.make_public()
