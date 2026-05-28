"""Добавляет корень репозитория в sys.path при запуске скриптов напрямую."""

from __future__ import annotations

import sys
from pathlib import Path

from slug_detection.paths import REPO_ROOT


def ensure_repo_on_sys_path(repo_root: Path | None = None) -> Path:
    root = (repo_root or REPO_ROOT).resolve()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def setup_path_from_file(caller_file: str, *, levels_up: int = 2) -> Path:
    """Добавляет корень репозитория в sys.path по пути вызывающего скрипта."""
    root = Path(caller_file).resolve().parents[levels_up]
    return ensure_repo_on_sys_path(root)
