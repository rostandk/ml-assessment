from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from depop.data import load_dataset_tsv


def test_load_dataset_tsv_valid(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "product_id": ["1"],
            "description": ["Test"],
            "image_url": ["http://example.com/img.jpg"],
            "label": [1],
            "yes_count": [5],
            "no_count": [2],
        }
    )
    path = tmp_path / "train.tsv"
    df.to_csv(path, sep="\t", index=False)
    loaded = load_dataset_tsv(path)
    assert "label_confidence" in loaded.columns
    assert pytest.approx(loaded.loc[0, "label_confidence"], 0.01) == (5 - 2) / 7


def test_load_dataset_tsv_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"product_id": ["1"]})
    path = tmp_path / "train.tsv"
    df.to_csv(path, sep="\t", index=False)
    with pytest.raises(ValueError):
        load_dataset_tsv(path)
