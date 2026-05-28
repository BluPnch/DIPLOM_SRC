"""Корень репозитория и стандартные каталоги проекта."""

from pathlib import Path

# slug_detection/paths.py → родитель slug_detection/ = корень репозитория
REPO_ROOT = Path(__file__).resolve().parent.parent

DETECTORS_DIR = REPO_ROOT / "detectors"
RUNS_DIR = REPO_ROOT / "runs"
DATASET_DIR = REPO_ROOT / "dataset"
INVESTIGATION_DIR = REPO_ROOT / "investigation"
SCRIPTS_DIR = REPO_ROOT / "scripts"
