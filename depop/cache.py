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
