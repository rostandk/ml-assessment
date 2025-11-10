from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from depop.artifacts import ArtifactManager
from depop.data import SFTDataset
from depop.media import ImageStore
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
    settings.gcs.enabled = False  # avoid accidental uploads in tests
    return settings


def test_image_store_ensure_all(monkeypatch, tmp_settings, tmp_path: Path):
    store = ImageStore(tmp_path / "cache" / "images", gcs_bucket="test-bucket")

    def fake_download(self, url: str, **kwargs):
        p = (self.cache_dir / "example.com" / "a" / "b.jpg")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"ok")
        return p

    def fake_upload(self, local: Path, url: str) -> str:
        return f"https://storage.googleapis.com/test-bucket/images/example.com/a/b.jpg"

    monkeypatch.setattr(ImageStore, "download", fake_download)
    monkeypatch.setattr(ImageStore, "upload", fake_upload)

    recs = store.ensure_all(["https://example.com/a/b.jpg"])  # type: ignore[list-item]
    assert len(recs) == 1
    assert recs[0]["public_url"].startswith("https://storage.googleapis.com/test-bucket")


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

    artifact_manager = ArtifactManager(tmp_settings)
    trainer = QwenTrainer(tmp_settings, artifact_manager)
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

    store = ImageStore(tmp_settings.paths.cache_dir, gcs_bucket="test-bucket")
    monkeypatch.setattr("depop.media.ImageStore.download", lambda self, url: None)
    runner = InferenceRunner(tmp_settings, store)
    df = pd.DataFrame({"product_id": ["1"], "description": ["desc"], "image_url": [""], "label": [1]})
    preds = runner.predict(df)
    assert len(preds) == 1
    assert bool(preds.loc[0, "is_spam_pred"]) is True


def test_artifact_manager_save_manifest(monkeypatch, tmp_settings, tmp_path: Path):
    artifact_manager = ArtifactManager(tmp_settings)
    calls: list[tuple[Path, str]] = []

    def fake_upload(path: Path, prefix: str) -> None:
        calls.append((path, prefix))

    monkeypatch.setattr(artifact_manager, "_upload_file", fake_upload)  # type: ignore[attr-defined]
    df = pd.DataFrame([{"image_url": "url", "downloaded": True, "uploaded": True}])
    output = artifact_manager.save_manifest(df)
    assert output.exists()
    assert output.name == "image_manifest.csv"
    assert calls and calls[0][0] == output


def test_artifact_manager_upload_model_dir(monkeypatch, tmp_settings, tmp_path: Path):
    artifact_manager = ArtifactManager(tmp_settings)
    model_dir = tmp_path / "model"
    (model_dir / "sub").mkdir(parents=True)
    file_a = model_dir / "config.json"
    file_b = model_dir / "sub" / "weights.bin"
    file_a.write_text("{}")
    file_b.write_bytes(b"0")

    uploaded: list[Path] = []

    def fake_upload(path: Path, prefix: str) -> None:
        uploaded.append(path)

    monkeypatch.setattr(artifact_manager, "_upload_file", fake_upload)  # type: ignore[attr-defined]
    artifact_manager.upload_model_dir(model_dir)
    assert set(uploaded) == {file_a, file_b}
