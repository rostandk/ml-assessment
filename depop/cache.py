from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from .media import MediaCache
from .settings import NotebookSettings


class CacheManager:
    """Settings-aware cache orchestration for the notebook."""

    def __init__(self, settings: NotebookSettings):
        self.settings = settings
        self.media = MediaCache(
            settings.paths.cache_dir,
            zip_url=settings.cache.zip_url,
            zip_path=settings.paths.cache_zip_path,
            min_images=settings.cache.min_images,
        )

    def ensure(self, urls: Sequence[str]) -> pd.DataFrame:
        self.media.bootstrap_from_zip()
        results = self.media.download_many(urls)
        return pd.DataFrame(results)

    def publish_if_enabled(self) -> Path | None:
        if not self.settings.cache.publish_zip:
            return None
        return self.media.publish_zip(self.settings.paths.publish_cache_zip_path)

    def sync_to_gcs(self, **kwargs) -> None:
        self.media.sync_to_gcs(**kwargs)

    @property
    def media_cache(self) -> MediaCache:
        return self.media

    # ------------------------------------------------------------------
    # Auto-persist and manifest wiring

    def auto_persist(self) -> None:
        """Persist cache to GCS if configured in settings."""
        if not self.settings.cache.gcs_enabled:
            return
        self.media.sync_to_gcs(
            bucket=self.settings.cache.gcs_bucket,
            prefix=self.settings.cache.gcs_prefix,
            public=self.settings.cache.gcs_public,
        )

    def write_manifest(self, status_df, out_path: Path) -> Path:
        """Create a manifest mapping original URL -> filename -> public URL if available."""
        import pandas as pd
        from .media import MediaCache

        rows = []
        base_url = None
        if self.settings.cache.gcs_enabled and self.settings.cache.gcs_public:
            base_url = f"https://storage.googleapis.com/{self.settings.cache.gcs_bucket}/{self.settings.cache.gcs_prefix.strip('/')}/"
        for rec in status_df.to_dict("records"):
            url = rec.get("image_url", "")
            if not url:
                continue
            filename = MediaCache.url_to_cache_filename(url)
            public_url = base_url + filename if base_url else ""
            rows.append({
                "image_url": url,
                "filename": filename,
                "public_url": public_url,
                "downloaded": bool(rec.get("downloaded", False)),
            })
        df = pd.DataFrame(rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        return out_path
