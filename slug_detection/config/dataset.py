"""Общее разрешение путей data.yaml и сплитов (совместимо с форматом Ultralytics)."""

from __future__ import annotations

from pathlib import Path

import yaml

from slug_detection.paths import DETECTORS_DIR, REPO_ROOT

DATA_YAML_CANDIDATES = [
    REPO_ROOT / "dataset" / "data.yaml",
    DETECTORS_DIR / "yolo" / "dataset" / "data.yaml",
    DETECTORS_DIR / "yolo26" / "dataset" / "data.yaml",
    DETECTORS_DIR / "yolo12" / "dataset" / "data.yaml",
    DETECTORS_DIR / "yolo11" / "dataset" / "data.yaml",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_data_yaml() -> Path:
    for candidate in DATA_YAML_CANDIDATES:
        if candidate.is_file():
            return candidate
    tried = "\n".join(f"  - {p}" for p in DATA_YAML_CANDIDATES)
    raise FileNotFoundError(
        "Не найден файл data.yaml. Искали:\n"
        f"{tried}\n"
        "Создайте YAML или положите датасет в один из этих каталогов."
    )


def load_data_cfg(yaml_path: Path | None = None) -> tuple[dict, Path]:
    path = yaml_path if yaml_path is not None else find_data_yaml()
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f), path.resolve()


def class_names_from_cfg(data_cfg: dict) -> list:
    names = data_cfg["names"]
    if isinstance(names, dict):

        def sort_key(k):
            try:
                return int(k)
            except (TypeError, ValueError):
                return str(k)

        return [names[k] for k in sorted(names, key=sort_key)]
    return list(names)


def resolve_split_images_dir(data_cfg: dict, split_key: str, yaml_path: Path) -> Path:
    yaml_path = yaml_path.resolve()
    yaml_parent = yaml_path.parent
    root = yaml_parent
    if data_cfg.get("path"):
        root = (yaml_parent / Path(str(data_cfg["path"]).strip())).resolve()
    rel = Path(str(data_cfg[split_key]).strip())
    if rel.is_absolute():
        return rel.resolve()
    return (root / rel).resolve()


def iter_image_paths(images_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in IMAGE_EXTS:
        paths.extend(images_dir.glob(f"*{ext}"))
        paths.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted(set(paths))


def yolo_labels_dir(images_dir: Path, split: str) -> Path:
    return images_dir.parent.parent / "labels" / split


def _yolo_best_scan_roots(project_root: Path) -> list[Path]:
    """Каталоги, где Ultralytics и ручные прогоны часто кладут runs/."""
    out: list[Path] = []
    for sub in ("runs/detect", "runs", "detectors/yolo/runs"):
        p = (project_root / Path(sub)).resolve()
        if p.is_dir():
            out.append(p)
    return out


def find_latest_yolo_best_weights(project_root: Path | None = None) -> Path | None:
    root = project_root or REPO_ROOT
    candidates: list[Path] = []
    for base in _yolo_best_scan_roots(root):
        candidates.extend(base.glob("**/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def ultralytics_dataset_root(data_cfg: dict, yaml_file: Path | str) -> Path:
    """
    Корень YOLO-датасета (каталог, где есть images/train, images/val, …).

    Если в YAML ошибочный path (напр. `./dataset`, при том что сам yaml уже лежит в `dataset/`),
    сначала проверяем каталог файла конфигурации, затем path из YAML.
    """
    yaml_file = Path(yaml_file).resolve()
    yaml_dir = yaml_file.parent
    train_rel = Path(str(data_cfg["train"]).strip())

    roots: list[Path] = []
    raw_path = data_cfg.get("path")
    if raw_path is not None and str(raw_path).strip() not in ("", "."):
        roots.append((yaml_dir / Path(str(raw_path).strip())).resolve())
    roots.append(yaml_dir.resolve())

    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        train_dir = root / train_rel
        if train_dir.is_dir():
            return root

    tried = ", ".join(str(r) for r in roots)
    raise FileNotFoundError(
        "Не удалось найти каталог данных для Ultralytics:\n"
        f"Ожидалась папка {train_rel} относительно одного из: {tried}"
    )


def export_ultralytics_yaml_with_absolute_path(
    source_yaml_path: Path | str, destination: Path | str
) -> Path:
    """Пишет копию data.yaml с абсолютным `path:` — чтобы не ломался resolve с глобальным Ultralytics datasets dir."""
    source_yaml_path = Path(source_yaml_path)
    cfg, yaml_path_loaded = load_data_cfg(source_yaml_path)
    dataset_root = ultralytics_dataset_root(cfg, yaml_path_loaded)

    absolute_cfg = dict(cfg)
    absolute_cfg["path"] = str(dataset_root.resolve())

    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        yaml.safe_dump(absolute_cfg, f, sort_keys=False, allow_unicode=True)
    return dst.resolve()


def resolve_yolo_best_weights(project_root: Path | None = None) -> Path | None:
    """Сначала типичный путь detect/train, иначе самый свежий best.pt среди известных runs/."""
    root = project_root or REPO_ROOT
    primary = root / "runs" / "detect" / "train" / "weights" / "best.pt"
    if primary.is_file():
        return primary
    alt = root / "detectors" / "yolo" / "runs" / "train" / "weights" / "best.pt"
    if alt.is_file():
        return alt
    return find_latest_yolo_best_weights(root)
