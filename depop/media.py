from __future__ import annotations

import concurrent.futures as cf
import logging
import random
import shutil
import time
import urllib.request
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Optional, Sequence

import requests

LOGGER = logging.getLogger(__name__)

LEGACY_PREFIX = "https://pictures.depop.com/"
MEDIA_PREFIX = "https://media-photos.depop.com/"


class MediaCache:
    """Handle image caching, bootstrap, and publishing."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        zip_url: Optional[str] = None,
        zip_path: Optional[Path] = None,
        min_images: int = 1,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.zip_url = zip_url
        self.zip_path = zip_path or self.cache_dir.parent / "images_cache.zip"
        self.min_images = max(min_images, 1)

    # ------------------------------------------------------------------
    # Filename helpers

    @staticmethod
    def normalize_url(url: str | None) -> str | None:
        if not url:
            return None
        url = url.strip()
        if url.startswith(LEGACY_PREFIX):
            return MEDIA_PREFIX + url[len(LEGACY_PREFIX) :]
        return url

    @classmethod
    def url_to_cache_filename(cls, url: str) -> str:
        normalised = cls.normalize_url(url)
        if normalised is None:
            raise ValueError("Cannot hash an empty URL")
        return f"{sha256(normalised.encode('utf-8')).hexdigest()}.jpg"

    @classmethod
    def local_path(cls, url: str, cache_dir: Path | str) -> Path:
        return Path(cache_dir) / cls.url_to_cache_filename(url)

    # ------------------------------------------------------------------
    # Download logic

    def download_image(
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

    def download_many(self, urls: Sequence[str], *, max_workers: int = 8) -> list[dict[str, object]]:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        def worker(url: str) -> dict[str, object]:
            path = self.download_image(url)
            return {"image_url": url, "downloaded": bool(path and path.exists())}

        results: list[dict[str, object]] = []
        with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for record in executor.map(worker, urls):
                results.append(record)
        return results

    # ------------------------------------------------------------------
    # Bootstrap/publish

    def bootstrap_from_zip(self) -> None:
        if not self.zip_url:
            return
        existing = sum(1 for _ in self.cache_dir.glob("*.jpg"))
        if existing >= self.min_images:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.zip_path
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            LOGGER.info("Bootstrapping cache from %s", self.zip_url)
            with urllib.request.urlopen(self.zip_url, timeout=60) as response, open(zip_path, "wb") as handle:
                shutil.copyfileobj(response, handle)
            with zipfile.ZipFile(zip_path, "r") as archive:
                for member in archive.infolist():
                    self._safe_extract(archive, member)
        except Exception as exc:  # pragma: no cover - network dependent
            LOGGER.warning("Cache bootstrap failed: %s", exc)

    def publish_zip(self, output_path: Path | None = None) -> Path:
        output = Path(output_path or self.zip_path)
        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for image_file in self.cache_dir.glob("*.jpg"):
                archive.write(image_file, image_file.name)
        return output

    def _safe_extract(self, archive: zipfile.ZipFile, member: zipfile.ZipInfo) -> None:
        target = (self.cache_dir / member.filename).resolve()
        cache_root = self.cache_dir.resolve()
        if not str(target).startswith(str(cache_root)):
            LOGGER.warning("Skipping suspicious zip member: %s", member.filename)
            return
        archive.extract(member, cache_root)

    # ------------------------------------------------------------------
    # GCS sync

    def iter_files(self, patterns: Iterable[str] = ("*.jpg", "*.jpeg", "*.png")) -> Iterable[Path]:
        for pattern in patterns:
            yield from self.cache_dir.rglob(pattern)

    def sync_to_gcs(
        self,
        *,
        bucket: str,
        prefix: str = "images",
        cache_control: str = "public, max-age=31536000",
        public: bool = False,
        threads: int = 16,
        dry_run: bool = False,
    ) -> None:
        from google.cloud import storage  # pragma: no cover - optional dependency

        files = list(self.iter_files())
        if not files:
            LOGGER.warning("No image files found under %s", self.cache_dir)
            return

        LOGGER.info("Preparing to upload %d files to gs://%s/%s", len(files), bucket, prefix)
        client = storage.Client()
        bucket_obj = client.bucket(bucket)

        def upload(path: Path) -> Optional[str]:
            blob_name = f"{prefix.rstrip('/')}/{path.name}"
            if dry_run:
                LOGGER.info("would upload %s -> gs://%s/%s", path, bucket, blob_name)
                return None
            blob = bucket_obj.blob(blob_name)
            blob.cache_control = cache_control
            blob.upload_from_filename(str(path))
            if public:
                blob.make_public()
                return blob.public_url
            return None

        uploaded = 0
        failed = 0
        sample_url: Optional[str] = None
        with cf.ThreadPoolExecutor(max_workers=threads) as executor:
            future_map = {executor.submit(upload, path): path for path in files}
            for future in cf.as_completed(future_map):
                path = future_map[future]
                try:
                    public_url = future.result()
                    uploaded += 1
                    if sample_url is None and public_url:
                        sample_url = public_url
                except Exception as exc:  # pragma: no cover
                    failed += 1
                    LOGGER.error("Failed to upload %s: %s", path, exc)
        LOGGER.info("Uploaded %d files (%d failed)", uploaded, failed)
        if public and sample_url:
            LOGGER.info("Example public URL: %s", sample_url)
