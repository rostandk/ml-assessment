from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from .media import MediaCache
from .settings import NotebookSettings

REQUIRED_COLUMNS = ["product_id", "description", "image_url", "label", "yes_count", "no_count"]


def load_dataset_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df[REQUIRED_COLUMNS].copy()
    df["product_id"] = df["product_id"].astype(str)
    if df["product_id"].duplicated().any():
        raise ValueError("Duplicate product_id values detected")
    df["label"] = df["label"].astype(int)
    for col in ("yes_count", "no_count"):
        df[col] = df[col].fillna(0).astype(int)
        if (df[col] < 0).any():
            raise ValueError(f"Negative values found in {col}")
    total_votes = df["yes_count"] + df["no_count"]
    with np.errstate(divide="ignore", invalid="ignore"):
        confidence = (df["yes_count"] - df["no_count"]) / np.where(total_votes == 0, np.nan, total_votes)
    df["label_confidence"] = confidence.fillna(0.0)
    return df


class DataModule:
    def __init__(self, settings: NotebookSettings):
        self.settings = settings

    def load_training_dataframe(self) -> pd.DataFrame:
        return load_dataset_tsv(self.settings.paths.data_train)

    def load_test_dataframe(self) -> pd.DataFrame:
        path = self.settings.paths.data_test
        if Path(path).exists():
            return load_dataset_tsv(path)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    def train_val_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_split, val_split = train_test_split(
            df,
            test_size=0.1,
            stratify=df["label"],
            random_state=self.settings.seed,
        )
        return train_split, val_split


class BaselineModel:
    def __init__(self, settings: NotebookSettings):
        self.settings = settings

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict[str, Any]:
        vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
        X_train = vectorizer.fit_transform(train_df["description"])
        X_val = vectorizer.transform(val_df["description"])

        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(X_train, train_df["label"])

        preds = clf.predict(X_val)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_df["label"], preds, average="macro", zero_division=0
        )
        accuracy = accuracy_score(val_df["label"], preds)
        return {
            "accuracy": float(accuracy),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
            "classification_report": classification_report(val_df["label"], preds, digits=3),
        }


@dataclass
class SFTDataset:
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    train_path: Path
    val_path: Path
    val_labels: list[int]


class SFTDatasetBuilder:
    def __init__(self, settings: NotebookSettings, media_cache: MediaCache):
        self.settings = settings
        self.media_cache = media_cache

    def build(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> SFTDataset:
        rows_train = [self._build_record(row) for row in train_df.to_dict("records")]
        rows_val = [self._build_record(row) for row in val_df.to_dict("records")]

        self.settings.paths.sft_dir.mkdir(parents=True, exist_ok=True)
        train_path = self.settings.paths.sft_dir / "train.jsonl"
        val_path = self.settings.paths.sft_dir / "val.jsonl"
        train_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_train))
        val_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_val))

        val_labels = val_df["label"].astype(int).reset_index(drop=True).tolist()
        return SFTDataset(rows_train, rows_val, train_path, val_path, val_labels)

    def _build_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        url = record.get("image_url") or ""
        image_path = self.media_cache.download_image(url) if url else None
        user_content: list[dict[str, Any]] = []
        note = ""
        if image_path is not None and image_path.exists():
            user_content.append({"type": "image", "image": str(image_path)})
        else:
            note = "\n\nNote: image unavailable. Base your judgment on the text only."
        user_content.append(
            {
                "type": "text",
                "text": f"{self.settings.training.prompt_template}\n\nDescription: {record['description']}{note}",
            }
        )
        assistant_text = json.dumps(
            {
                "is_spam": bool(record.get("label", 0)),
                "confidence": 1.0,
                "reason": "Training label",
            },
            ensure_ascii=False,
        )
        return {
            "id": record.get("product_id"),
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "Respond with strict JSON."}]},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
            ],
        }
