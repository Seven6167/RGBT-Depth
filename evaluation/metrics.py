"""MAE, RMSE, GAME(0-3), and dataset-specific GT maps."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from inference.data import _drone_xml_points, _rgbtcc_json_points


DENSITY_HW = {"DroneRGBT": (64, 80), "RGBT-CC": (60, 80)}


def ground_truth_map(dataset: str, annotation: Path, rgb: Path,
                     original_hw: tuple[int, int]) -> np.ndarray:
    if dataset == "RGBT-CC":
        height, width = original_hw
        out_h, out_w = DENSITY_HW[dataset]
        target = np.zeros((out_h, out_w), dtype=np.float32)
        for x, y in _rgbtcc_json_points(annotation):
            if 0 <= x < width and 0 <= y < height:
                xi = min(max(int(np.floor(x * out_w / width)), 0), out_w - 1)
                yi = min(max(int(np.floor(y * out_h / height)), 0), out_h - 1)
                target[yi, xi] += 1.0
        return target

    if dataset != "DroneRGBT":
        raise ValueError(f"Unsupported dataset: {dataset}")
    import cv2
    from scipy.ndimage import gaussian_filter

    image = cv2.imread(str(rgb), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read RGB image: {rgb}")
    height, width = image.shape[:2]
    out_h, out_w = DENSITY_HW[dataset]
    points = _drone_xml_points(annotation)
    impulses = np.zeros((out_h, out_w), dtype=np.float32)
    for x, y in points:
        xi = int(round(x * out_w / max(width, 1)))
        yi = int(round(y * out_h / max(height, 1)))
        if 0 <= xi < out_w and 0 <= yi < out_h:
            impulses[yi, xi] += 1.0
    if not points:
        return impulses
    sigma = max(8.0 * out_w / max(width, 1), 0.5)
    target = gaussian_filter(impulses, sigma=sigma, mode="constant")
    total = target.sum()
    if total > 1e-8:
        target *= len(points) / total
    return target.astype(np.float32)


def game_per_image(prediction: np.ndarray, target: np.ndarray) -> list[float]:
    pred = np.asarray(prediction, dtype=np.float64)
    gt = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or pred.shape != gt.shape:
        raise ValueError("GAME expects matched HxW density maps")
    if not (np.isfinite(pred).all() and np.isfinite(gt).all()):
        raise ValueError("Density map contains non-finite values")
    height, width = pred.shape
    games = []
    for level in range(4):
        cells = 2 ** level
        error = 0.0
        for row in range(cells):
            y0, y1 = row * height // cells, (row + 1) * height // cells
            for column in range(cells):
                x0, x1 = column * width // cells, (column + 1) * width // cells
                error += abs(float(pred[y0:y1, x0:x1].sum())
                             - float(gt[y0:y1, x0:x1].sum()))
        games.append(error)
    return games


def aggregate(rows: list[dict], dataset: str) -> dict:
    if not rows:
        raise ValueError("No evaluated samples")
    if dataset == "DroneRGBT":
        signed = np.asarray(
            [row["predicted_count"] - row["ground_truth"] for row in rows],
            dtype=np.float64,
        )
        games = np.asarray([row["game"] for row in rows], dtype=np.float64)
        result = {
            "num_images": len(rows),
            "MAE": float(np.abs(signed).mean()),
            "RMSE": float(np.sqrt(np.square(signed).mean())),
            **{f"GAME({level})": float(games[:, level].mean()) for level in range(4)},
        }
    elif dataset == "RGBT-CC":
        abs_error = squared_error = 0.0
        games = [0.0] * 4
        for row in rows:
            error = row["predicted_count"] - row["ground_truth"]
            abs_error += abs(error)
            squared_error += error ** 2
            games = [total + value for total, value in zip(games, row["game"])]
        count = len(rows)
        result = {
            "num_images": count,
            "MAE": abs_error / count,
            "RMSE": math.sqrt(squared_error / count),
            **{f"GAME({level})": games[level] / count for level in range(4)},
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not math.isclose(result["MAE"], result["GAME(0)"], abs_tol=1e-4):
        raise RuntimeError("MAE and GAME(0) differ")
    return result
