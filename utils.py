"""Utility helpers for the Colab keyword spam workflow."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import random
import time
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Optional, Sequence

import requests

LOGGER = logging.getLogger(__name__)

LEGACY_PREFIX = "https://pictures.depop.com/"
MEDIA_PREFIX = "https://media-photos.depop.com/"


def normalize_url(url: str | None) -> str | None:
    """Return the canonical form of a Depop media URL."""

    if not url:
        return None
    url = url.strip()
    if url.startswith(LEGACY_PREFIX):
        return MEDIA_PREFIX + url[len(LEGACY_PREFIX) :]
    return url


def url_to_cache_filename(url: str) -> str:
    """Map a media URL to its hashed cache filename (legacy-compatible)."""

    normalised = normalize_url(url)
    if normalised is None:
        raise ValueError("Cannot hash an empty URL")
    return f"{sha256(normalised.encode('utf-8')).hexdigest()}.jpg"


def local_image_path(url: str, cache_dir: Path | str) -> Path:
    """Return the expected local path for *url* inside *cache_dir*."""

    return Path(cache_dir) / url_to_cache_filename(url)


def download_image(
    url: str,
    cache_dir: Path | str,
    *,
    timeout: int = 15,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    headers: Optional[dict[str, str]] = None,
) -> Optional[Path]:
    """Download *url* into *cache_dir* if missing and return the local path."""

    normalised = normalize_url(url)
    if not normalised:
        return None

    destination = Path(cache_dir) / url_to_cache_filename(normalised)
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


def gcs_blob_path(url: str, *, prefix: str = "images/") -> str:
    """Return the blob path used when syncing an image to GCS."""

    return f"{prefix.rstrip('/')}/{url_to_cache_filename(url)}"


def gcs_public_url(url: str, *, bucket_base_url: str, prefix: str = "images/") -> str:
    """Return the public HTTPS URL for an uploaded image."""

    base = bucket_base_url.rstrip("/")
    blob = gcs_blob_path(url, prefix=prefix)
    return f"{base}/{blob.lstrip('/')}"


def iter_image_files(source: Path | str, patterns: Iterable[str] = ("*.jpg", "*.jpeg", "*.png")) -> Iterable[Path]:
    """Yield image files from *source* matching the provided glob patterns."""

    root = Path(source)
    for pattern in patterns:
        yield from root.rglob(pattern)


def download_images(
    urls: Sequence[str],
    cache_dir: Path | str,
    *,
    max_workers: int = 8,
    timeout: int = 15,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
    headers: Optional[dict[str, str]] = None,
) -> list[dict[str, object]]:
    """Download many URLs concurrently, returning status records."""

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    def worker(url: str) -> dict[str, object]:
        path = download_image(
            url,
            cache_path,
            timeout=timeout,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            headers=headers,
        )
        return {
            "image_url": url,
            "downloaded": bool(path and path.exists()),
        }

    results: list[dict[str, object]] = []
    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for record in executor.map(worker, urls):
            results.append(record)
    return results


def sync_images_to_gcs(
    source: Path | str,
    *,
    bucket: str,
    prefix: str = "images",
    cache_control: str = "public, max-age=31536000",
    public: bool = False,
    threads: int = 16,
    dry_run: bool = False,
) -> None:
    """Upload images from *source* to a GCS bucket."""

    from google.cloud import storage  # imported lazily to keep base deps light

    source_path = Path(source)
    if not source_path.is_dir():
        raise ValueError(f"Source directory {source_path} does not exist")

    files = list(iter_image_files(source_path))
    if not files:
        LOGGER.warning("No image files found under %s", source_path)
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
            except Exception as exc:  # noqa: BLE001
                failed += 1
                LOGGER.error("Failed to upload %s: %s", path, exc)

    LOGGER.info("Uploaded %d files (%d failed)", uploaded, failed)
    if public and sample_url:
        LOGGER.info("Example public URL: %s", sample_url)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Utility CLI for keyword spam assets")
    sub = parser.add_subparsers(dest="command", required=True)

    sync_parser = sub.add_parser("sync-gcs", help="Upload images to a GCS bucket")
    sync_parser.add_argument("--source", required=True, help="Path to cache/images directory")
    sync_parser.add_argument("--bucket", required=True, help="GCS bucket name (without gs://)")
    sync_parser.add_argument("--prefix", default="images", help="Prefix inside the bucket (default: images)")
    sync_parser.add_argument("--cache-control", default="public, max-age=31536000", help="Cache-Control header")
    sync_parser.add_argument("--public", action="store_true", help="Make uploaded objects public")
    sync_parser.add_argument("--threads", type=int, default=16, help="Parallel upload workers (default: 16)")
    sync_parser.add_argument("--dry-run", action="store_true", help="List actions without uploading")

    return parser.parse_args(argv)


def _cli(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.command == "sync-gcs":
        sync_images_to_gcs(
            args.source,
            bucket=args.bucket,
            prefix=args.prefix,
            cache_control=args.cache_control,
            public=args.public,
            threads=args.threads,
            dry_run=args.dry_run,
        )
    else:  # pragma: no cover - safety net for future commands
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":  # pragma: no cover
    _cli()
