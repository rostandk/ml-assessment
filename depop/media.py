from __future__ import annotations

import concurrent.futures as cf
import logging
import random
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import urlparse, urlunparse

import requests

LOGGER = logging.getLogger(__name__)

class ImageStore:
    """Deterministic, path-based image store with public GCS upload."""

    LEGACY_HOST_REMAP = {
        "pictures.depop.com": "media-photos.depop.com",
    }

    def __init__(self, cache_dir: Path, *, gcs_bucket: str, gcs_images_prefix: str = "images") -> None:
        self.cache_dir = Path(cache_dir)
        self.gcs_bucket = gcs_bucket
        self.gcs_images_prefix = gcs_images_prefix.strip("/")

    # ------------------------------------------------------------------
    # Filename helpers

    @classmethod
    def normalize_url(cls, url: str | None) -> str | None:
        if not url:
            return None
        raw = url.strip()
        parsed = urlparse(raw if "://" in raw else f"https://{raw.lstrip('/')}")
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or ""
        remapped = cls.LEGACY_HOST_REMAP.get(netloc.lower())
        netloc = remapped or netloc
        if not netloc:
            return raw  # fallback to original string
        rebuilt = urlunparse(
            (
                scheme,
                netloc,
                parsed.path or "",
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
        return rebuilt

    @classmethod
    def url_to_relative_path(cls, url: str) -> Path:
        """Map a URL to a safe, deterministic relative path host/path.

        Example: https://media-photos.depop.com/b0/a/b/P0.jpg ->
                 media-photos.depop.com/b0/a/b/P0.jpg
        Query/fragment removed; unsafe chars replaced with '_'.
        """
        from urllib.parse import urlparse

        norm = cls.normalize_url(url)
        if not norm:
            raise ValueError("Empty URL")
        u = urlparse(norm)
        host = (u.netloc or "unknown-host").replace("..", "_")
        path = (u.path or "/unknown.jpg").lstrip("/")
        # sanitize path
        safe = path.replace("..", "_").replace(" ", "_")
        if not safe:
            safe = "unknown.jpg"
        if "/" in safe:
            rel = Path(host) / Path(safe)
        else:
            rel = Path(host) / safe
        # default extension
        if not rel.suffix:
            rel = rel.with_suffix(".jpg")
        return rel

    @classmethod
    def local_path(cls, url: str, cache_dir: Path | str) -> Path:
        return Path(cache_dir) / cls.url_to_relative_path(url)

    # ------------------------------------------------------------------
    # Download logic

    def download(
        self,
        url: str,
        *,
        timeout: int = 15,
        max_retries: int = 3,
        backoff_factor: float = 1.5,
        headers: Optional[dict[str, str]] = None,
    ) -> Optional[Path]:
        normalised = self.normalize_url(url)
        if not normalised:
            return None
        destination = self.local_path(normalised, self.cache_dir)
        if destination.exists():
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        request_headers = headers or {
            "User-Agent": "Mozilla/5.0 (compatible; KeywordSpamBot/1.0)",
        }

        for attempt in range(max_retries):
            try:
                response = requests.get(normalised, timeout=timeout, headers=request_headers)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    LOGGER.warning("Skipping %s (content-type=%s)", normalised, content_type)
                    return None
                destination.write_bytes(response.content)
                return destination
            except requests.RequestException as exc:
                sleep_for = backoff_factor ** attempt * random.uniform(0.8, 1.2)
                LOGGER.debug("Download failed for %s (%s), retrying in %.2fs", normalised, exc, sleep_for)
                time.sleep(sleep_for)
        LOGGER.warning("Giving up on %s after %d retries", normalised, max_retries)
        return None

    def ensure_all(self, urls: Sequence[str], *, max_workers: int = 8) -> list[dict[str, object]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        def worker(url: str) -> dict[str, object]:
            local = self.download(url)
            downloaded = bool(local and Path(local).exists())
            public_url = ""
            uploaded = False
            if downloaded:
                try:
                    public_url = self.upload(local, url)
                    uploaded = bool(public_url)
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("Upload failed for %s: %s", url, exc)
            rel = str(self.url_to_relative_path(url))
            return {
                "image_url": url,
                "relative_path": rel,
                "local_path": str(local) if local else "",
                "public_url": public_url,
                "downloaded": downloaded,
                "uploaded": uploaded,
            }

        results: list[dict[str, object]] = []
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for record in executor.map(worker, urls):
                results.append(record)
        return results
    # ------------------------------------------------------------------
    # Upload to GCS (public)

    def upload(self, local_path: Path, url: str) -> str:
        from google.cloud import storage  # pragma: no cover - optional dependency

        client = storage.Client()
        bucket = client.bucket(self.gcs_bucket)
        rel = self.url_to_relative_path(url)
        blob_name = f"{self.gcs_images_prefix}/{str(rel).strip('/')}"
        blob = bucket.blob(blob_name)
        # idempotent skip when sizes match
        try:
            if blob.exists() and Path(local_path).exists():
                if blob.size == Path(local_path).stat().st_size:  # type: ignore[attr-defined]
                    return f"https://storage.googleapis.com/{self.gcs_bucket}/{blob_name}"
        except Exception:
            pass
        blob.cache_control = "public, max-age=31536000"
        blob.content_type = "image/jpeg"
        blob.upload_from_filename(str(local_path))
        blob.make_public()
        return f"https://storage.googleapis.com/{self.gcs_bucket}/{blob_name}"
