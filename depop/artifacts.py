from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict

import pandas as pd

from .settings import NotebookSettings


class ArtifactManager:
    """Persist predictions, metrics, and packaged artifacts."""

    def __init__(self, settings: NotebookSettings):
        self.settings = settings

    def save_predictions(self, df: pd.DataFrame) -> None:
        df.to_parquet(self.settings.paths.predictions_path, index=False)

    def save_metrics(self, metrics: Dict[str, object]) -> None:
        self.settings.paths.metrics_path.write_text(json.dumps(metrics, indent=2))

    def save_classification_report(self, report: str) -> None:
        self.settings.paths.classification_report_path.write_text(report)

    def package(self) -> Path:
        output = self.settings.paths.packaged_zip
        if output.exists():
            output.unlink()
        shutil.make_archive(str(output.with_suffix("")), "zip", self.settings.paths.artifacts_dir)
        return output
