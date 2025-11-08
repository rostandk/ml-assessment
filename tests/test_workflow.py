from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from depop.cache import CacheManager
from depop.data import SFTDataset
from depop.media import MediaCache
from depop.inference import InferenceRunner
from depop.settings import load_settings
from depop.training import QwenTrainer


@pytest.fixture
def tmp_settings(tmp_path: Path):
    overrides = {
        "repo_dir": tmp_path,
        "data_dir": tmp_path,
        "cache_dir": tmp_path / "cache" / "images",
        "artifacts_dir": tmp_path / "artifacts",
    }
    settings = load_settings(overrides)
    settings.paths.cache_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    return settings


def test_cache_manager_bootstrap(monkeypatch, tmp_settings):
    cache_manager = CacheManager(tmp_settings)
    called = {"bootstrap": False, "download": False}

    def fake_bootstrap(self):
        called["bootstrap"] = True

    def fake_download_many(self, urls, max_workers=8):
        called["download"] = True
        return [{"image_url": url, "downloaded": True} for url in urls]

    monkeypatch.setattr("depop.cache.MediaCache.bootstrap_from_zip", fake_bootstrap)
    monkeypatch.setattr("depop.cache.MediaCache.download_many", fake_download_many)

    df = cache_manager.ensure(["a", "b"])
    assert called["bootstrap"] and called["download"]
    assert len(df) == 2


def test_qwen_trainer_invokes_trainer(monkeypatch, tmp_settings, tmp_path: Path):
    dataset = SFTDataset(
        train_rows=[],
        val_rows=[],
        train_path=tmp_path / "train.jsonl",
        val_path=tmp_path / "val.jsonl",
        val_labels=[0],
    )
    dataset.train_path.write_text("{}\n")
    dataset.val_path.write_text("{}\n")

    class FakeModel:
        def to(self, *_):
            return self

        def eval(self):
            return self

        def save_pretrained(self, path: str):
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeTokenizer:
        def apply_chat_template(self, *_, **__):
            return "prompt"

        def batch_decode(self, *_, **__):
            return ["{}"]

        def save_pretrained(self, path: str):
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakeTrainer:
        def __init__(self, *_, **__):
            pass

        def train(self):
            return type("Result", (), {"metrics": {"loss": 0.0}})()

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def get_device_name(*_) -> str:
                return "CPU"

        @staticmethod
        def device(_):
            return "cpu"

    class FakeFastVisionModel:
        @staticmethod
        def from_pretrained(*_, **__):
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **__):
            return model

        @staticmethod
        def for_training(*_):
            return None

        @staticmethod
        def for_inference(*_):
            return None

    monkeypatch.setattr("depop.training.FastVisionModel", FakeFastVisionModel)
    monkeypatch.setattr("depop.training.UnslothVisionDataCollator", lambda *_, **__: None)
    monkeypatch.setattr("depop.training.SFTTrainer", FakeTrainer)
    monkeypatch.setattr("depop.training.SFTConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr("depop.training.EarlyStoppingCallback", lambda *_, **__: None)
    monkeypatch.setattr("depop.training.hf_load_dataset", lambda *_, **__: {"train": [], "validation": []})
    monkeypatch.setattr("depop.training.torch", FakeTorch)

    trainer = QwenTrainer(tmp_settings)
    summary = trainer.train(dataset)
    assert "train_result" in summary
    assert summary["model_dir"].exists()


def test_inference_runner(monkeypatch, tmp_settings, tmp_path: Path):
    class FakeModel:
        def to(self, *_):
            return self

        def eval(self):
            return self

        def generate(self, *_, **__):
            return ["{}"]

    class FakeTokenizer:
        def apply_chat_template(self, *_, **__):
            return "prompt"

        def batch_decode(self, *_, **__):
            return ["{\"is_spam\": true, \"confidence\": 0.9, \"reason\": \"spam\"}"]

        def __call__(self, *_, **__):
            class DummyTensor:
                def to(self, *_):
                    return self

            return {"input_ids": DummyTensor()}

    class FakeFastVisionModel:
        @staticmethod
        def from_pretrained(*_, **__):
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def for_inference(*_):
            return None

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def get_device_name(*_) -> str:
                return "CPU"

        @staticmethod
        def device(_):
            return "cpu"

        class no_grad:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return None

    monkeypatch.setattr("depop.inference.FastVisionModel", FakeFastVisionModel)
    monkeypatch.setattr("depop.inference.torch", FakeTorch)

    media_cache = tmp_settings.paths.cache_dir
    media_cache.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("depop.media.MediaCache.download_image", lambda self, url: None)
    runner = InferenceRunner(tmp_settings, MediaCache(tmp_settings.paths.cache_dir))
    df = pd.DataFrame({"product_id": ["1"], "description": ["desc"], "image_url": [""], "label": [1]})
    preds = runner.predict(df)
    assert len(preds) == 1
    assert bool(preds.loc[0, "is_spam_pred"]) is True
