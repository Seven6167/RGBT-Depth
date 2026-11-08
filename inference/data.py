"""Source-aligned RGB/TIR input and optional ground-truth protocols."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DATASET_SIZE_HW = {"DroneRGBT": (512, 640), "RGBT-CC": (480, 640)}


@dataclass(frozen=True)
class Sample:
    sample_id: str
    rgb: Path
    tir: Path
    gt: Path | None = None


def _require_dataset(dataset: str) -> None:
    if dataset not in DATASET_SIZE_HW:
        raise ValueError(f"Unsupported dataset: {dataset}")


def preprocess_pair(dataset: str, rgb_path: str | Path, tir_path: str | Path,
                    *, return_original_hw: bool = False):
    """Build the source-aligned six-channel model input."""
    _require_dataset(dataset)
    import cv2
    import torch

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    tir = cv2.imread(str(tir_path), cv2.IMREAD_COLOR)
    if rgb is None or tir is None:
        raise FileNotFoundError(f"Missing/undecodable RGB/TIR pair: {rgb_path}, {tir_path}")
    if dataset == "RGBT-CC" and rgb.shape[:2] != tir.shape[:2]:
        raise RuntimeError(
            f"Original RGB/TIR image shape mismatch: {rgb.shape[:2]} != {tir.shape[:2]}"
        )
    original_hw = tuple(map(int, rgb.shape[:2]))
    height, width = DATASET_SIZE_HW[dataset]
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    tensors = []
    for image in (rgb, tir):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        tensors.append((tensor - mean) / std)
    pair = torch.cat(tensors, dim=0).unsqueeze(0).contiguous()
    if tuple(pair.shape) != (1, 6, height, width) or not torch.isfinite(pair).all():
        raise RuntimeError("Invalid normalized RGB-T input tensor")
    return (pair, original_hw) if return_original_hw else pair


def _drone_xml_points(path: Path) -> list[tuple[float, float]]:
    from xml.etree import ElementTree as ET

    root = ET.parse(str(path)).getroot()
    points = []
    for point in root.findall("object/point"):
        x = point.find("x")
        y = point.find("y")
        if x is None or y is None or x.text is None or y.text is None:
            raise RuntimeError(f"Malformed DroneRGBT point annotation: {path}")
        points.append((float(x.text), float(y.text)))
    return points


def _rgbtcc_json_points(path: Path) -> list[tuple[float, float]]:
    import numpy as np

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "points" not in payload:
        raise RuntimeError(f"RGBT-CC annotation lacks 'points': {path}")
    points = np.asarray(payload["points"], dtype=np.float32)
    if points.size == 0:
        return []
    if points.ndim != 2 or points.shape[1] < 2:
        raise RuntimeError(f"Invalid RGBT-CC point array shape {points.shape}: {path}")
    if not np.isfinite(points[:, :2]).all():
        raise RuntimeError(f"Non-finite RGBT-CC point coordinate: {path}")
    return [(float(x), float(y)) for x, y in points[:, :2]]


def ground_truth_count(dataset: str, gt_path: str | Path,
                       original_hw: tuple[int, int] | None = None) -> int:
    """Return source-aligned point count; RGBT-CC filters out-of-image points."""
    _require_dataset(dataset)
    path = Path(gt_path)
    if dataset == "DroneRGBT":
        if path.suffix.lower() != ".xml":
            raise ValueError("DroneRGBT ground truth must be an XML file")
        return len(_drone_xml_points(path))
    if path.suffix.lower() != ".json":
        raise ValueError("RGBT-CC ground truth must be a JSON file")
    if original_hw is None:
        raise ValueError("RGBT-CC ground-truth parsing requires original image height/width")
    height, width = original_hw
    return sum(1 for x, y in _rgbtcc_json_points(path) if 0 <= x < width and 0 <= y < height)


def _safe_relative(root: Path, value: str) -> Path:
    if not value or Path(value).is_absolute():
        raise ValueError(f"Manifest path must be nonempty and relative: {value!r}")
    root = root.resolve(strict=True)
    candidate = (root / value).resolve(strict=True)
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"Manifest file escapes or is missing under input root: {value!r}")
    return candidate


def read_manifest(dataset: str, manifest: str | Path, input_root: str | Path) -> list[Sample]:
    """Read a minimal CSV or JSONL manifest using explicit sample identifiers."""
    _require_dataset(dataset)
    path, root = Path(manifest), Path(input_root)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        raise ValueError("Manifest must use .csv, .jsonl, or .ndjson")
    if not rows:
        raise ValueError(f"Empty input manifest: {path}")
    samples, seen = [], set()
    for row in rows:
        sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
        rgb = row.get("rgb")
        tir = row.get("tir") or row.get("thermal")
        gt = row.get("gt") or row.get("annotation")
        if not sample_id or not rgb or not tir:
            raise ValueError("Each manifest row requires sample_id/id, rgb, and tir/thermal")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample ID in manifest: {sample_id}")
        seen.add(sample_id)
        samples.append(Sample(sample_id, _safe_relative(root, str(rgb)),
                              _safe_relative(root, str(tir)),
                              _safe_relative(root, str(gt)) if gt else None))
    return samples


def _images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _index(paths: Iterable[Path], key_fn, label: str) -> dict[str, Path]:
    result = {}
    for path in paths:
        key = key_fn(path.stem)
        if not key:
            raise ValueError(f"Cannot derive sample ID for {label}: {path}")
        if key in result:
            raise ValueError(f"Ambiguous duplicate {label} sample ID {key}: {result[key]}, {path}")
        result[key] = path
    return result


def _first_dir(root: Path, names: tuple[str, ...], *, required: bool) -> Path | None:
    matches = [root / name for name in names if (root / name).is_dir()]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous input directories: {matches}")
    if matches:
        return matches[0]
    if required:
        raise FileNotFoundError(f"Expected one of these directories under {root}: {names}")
    return None


def discover_samples(dataset: str, input_root: str | Path) -> list[Sample]:
    """Pair directory inputs by canonical IDs, never by positional sorting."""
    _require_dataset(dataset)
    root = Path(input_root).resolve(strict=True)
    if dataset == "DroneRGBT":
        rgb_dir = _first_dir(root, ("RGB", "rgb"), required=True)
        tir_dir = _first_dir(root, ("Infrared", "TIR", "thermal", "tir"), required=True)
        gt_dir = _first_dir(root, ("GT_", "GT", "gt", "annotations"), required=False)
        rgb = _index(_images(rgb_dir), lambda s: s, "RGB")
        tir = _index(_images(tir_dir), lambda s: s[:-1] if s.endswith("R") else s, "TIR")
        gt = _index(sorted(gt_dir.glob("*.xml")) if gt_dir else [],
                    lambda s: s[:-1] if s.endswith("R") else s, "GT")
    else:
        rgb_dir = _first_dir(root, ("RGB", "rgb", "visible", "color"), required=True)
        tir_dir = _first_dir(root, ("TIR", "tir", "thermal", "Infrared", "ir"), required=True)
        gt_dir = _first_dir(root, ("GT", "GT_", "gt", "annotations"), required=False)

        def clean(stem: str) -> str:
            value = stem.lower()
            modalities = r"rgb|visible|color|thermal|infrared|tir|ir|gt|annotation"
            value = re.sub(rf"^({modalities})[_-]*", "", value)
            value = re.sub(r"^t[_-]+", "", value)
            value = re.sub(rf"[_-]*({modalities})$", "", value)
            value = re.sub(r"[_-]+t$", "", value)
            return value

        rgb = _index(_images(rgb_dir), clean, "RGB")
        tir = _index(_images(tir_dir), clean, "TIR")
        gt = _index(sorted(gt_dir.glob("*.json")) if gt_dir else [], clean, "GT")
    if set(rgb) != set(tir):
        missing_tir, missing_rgb = sorted(set(rgb) - set(tir)), sorted(set(tir) - set(rgb))
        raise RuntimeError(
            f"RGB/TIR sample-ID sets differ; missing TIR={missing_tir[:10]}, "
            f"missing RGB={missing_rgb[:10]}"
        )
    unknown_gt = set(gt) - set(rgb)
    if unknown_gt:
        raise RuntimeError(f"GT contains unmatched sample IDs: {sorted(unknown_gt)[:10]}")
    if not rgb:
        raise FileNotFoundError(f"No paired RGB/TIR images found under {root}")
    return [Sample(key, rgb[key], tir[key], gt.get(key)) for key in sorted(rgb)]


def single_sample(dataset: str, rgb: str | Path, tir: str | Path,
                  gt: str | Path | None = None, sample_id: str | None = None) -> Sample:
    _require_dataset(dataset)
    rgb_path, tir_path = Path(rgb).resolve(strict=True), Path(tir).resolve(strict=True)
    gt_path = Path(gt).resolve(strict=True) if gt else None
    identifier = sample_id or rgb_path.stem
    if not identifier.strip():
        raise ValueError("sample_id must not be empty")
    return Sample(identifier.strip(), rgb_path, tir_path, gt_path)
