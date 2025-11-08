from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .settings import NotebookSettings


class RepoManager:
    """Clone and activate the project repository inside Colab runtimes."""

    def __init__(self, settings: NotebookSettings):
        self.settings = settings

    def ensure(self) -> Path:
        repo_dir = self.settings.paths.repo_dir
        if not self.settings.is_colab:
            return repo_dir
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1", self.settings.repo_url, str(repo_dir)],
                check=True,
            )
        else:
            subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
        return repo_dir

    def activate(self) -> None:
        repo_dir = self.settings.paths.repo_dir
        os.chdir(repo_dir)
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
