from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from unittest.mock import Mock

import pytest
import requests

from utils import (
    download_image,
    download_images,
    gcs_blob_path,
    gcs_public_url,
    local_image_path,
    normalize_url,
    url_to_cache_filename,
)


def test_normalize_url_handles_legacy_prefix() -> None:
    legacy = "https://pictures.depop.com/products/123.jpg"
    expected = "https://media-photos.depop.com/products/123.jpg"
    assert normalize_url(legacy) == expected
    assert normalize_url(expected) == expected


def test_url_to_cache_filename_matches_sha256() -> None:
    url = "https://media-photos.depop.com/products/2023/09/12/file.jpg"
    digest = sha256(url.encode("utf-8")).hexdigest()
    assert url_to_cache_filename(url) == f"{digest}.jpg"


def test_local_image_path_uses_cache_dir(tmp_path: Path) -> None:
    url = "https://example.com/foo.jpg"
    expected = tmp_path / url_to_cache_filename(url)
    assert local_image_path(url, tmp_path) == expected


def test_gcs_blob_helpers() -> None:
    url = "https://example.com/foo.jpg"
    blob = gcs_blob_path(url, prefix="images")
    assert blob.startswith("images/")
    public_url = gcs_public_url(url, bucket_base_url="https://storage.googleapis.com/ml-assesment")
    assert public_url.endswith(url_to_cache_filename(url))
    assert public_url.startswith("https://storage.googleapis.com/ml-assesment/")


def test_url_to_cache_filename_rejects_empty() -> None:
    with pytest.raises(ValueError):
        url_to_cache_filename("")


def test_download_image_success(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("utils.time.sleep", lambda *_: None)

    def fake_get(url: str, timeout: int, headers: dict[str, str]) -> Mock:
        assert "User-Agent" in headers
        response = Mock()
        response.content = b"data"
        response.headers = {"Content-Type": "image/jpeg"}
        response.raise_for_status = Mock()
        return response

    monkeypatch.setattr("utils.requests.get", fake_get)

    path = download_image("https://example.com/foo.jpg", tmp_path)
    assert path is not None
    assert path.exists()
    assert path.read_bytes() == b"data"


def test_download_image_retries_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("utils.time.sleep", lambda *_: None)

    attempts = {"count": 0}

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise requests.RequestException("boom")
        response = Mock()
        response.content = b"ok"
        response.headers = {"Content-Type": "image/png"}
        response.raise_for_status = Mock()
        return response

    monkeypatch.setattr("utils.requests.get", fake_get)

    path = download_image("https://example.com/bar.jpg", tmp_path, max_retries=5)
    assert attempts["count"] == 3
    assert path is not None
    assert path.read_bytes() == b"ok"


def test_download_image_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("utils.time.sleep", lambda *_: None)

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        raise requests.RequestException("fail")

    monkeypatch.setattr("utils.requests.get", fake_get)

    path = download_image("https://example.com/baz.jpg", tmp_path, max_retries=2)
    assert path is None


def test_download_image_rejects_non_image(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("utils.time.sleep", lambda *_: None)

    def fake_get(url: str, timeout: int, headers: dict[str, str]):
        response = Mock()
        response.content = b"<html>"
        response.headers = {"Content-Type": "text/html"}
        response.raise_for_status = Mock()
        return response

    monkeypatch.setattr("utils.requests.get", fake_get)

    path = download_image("https://example.com/not-image", tmp_path)
    assert path is None


def test_download_images_parallel(monkeypatch, tmp_path: Path) -> None:
    def fake_download(url: str, cache_dir: Path, **_: object) -> Path:
        path = tmp_path / f"{url.split('/')[-1]}.jpg"
        path.write_bytes(b"ok")
        return path

    monkeypatch.setattr("utils.download_image", fake_download)

    urls = [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]
    results = download_images(urls, tmp_path, max_workers=2)
    assert len(results) == len(urls)
    assert all(record["downloaded"] for record in results)
