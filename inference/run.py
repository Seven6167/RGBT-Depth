"""Unified fixed-depth and instance-adaptive RGB-T crowd-counting CLI."""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

from .checkpoint import load_model
from .data import (DATASET_SIZE_HW, Sample, discover_samples, ground_truth_count,
                   preprocess_pair, read_manifest, single_sample)


CSV_FIELDS = (
    "row_type", "dataset", "scope", "sample_id", "mode", "ground_truth",
    "predicted_count", "selected_depth", "count_error", "absolute_error",
    "relative_error_percent", "GAME(0)", "GAME(1)", "GAME(2)", "GAME(3)",
    "num_samples", "MAE", "RMSE", "params_m", "flops_g", "latency_ms",
    "fps", "checkpoint_sha256", "membership_sha256",
    "test_membership_sha256",
)


def _replace_results_csv(output_root: Path, dataset: str, rows: list[dict]) -> Path:
    output = (output_root.expanduser().absolute() / dataset)
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise FileExistsError(f"Invalid output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "results.csv"
    if target.is_symlink():
        raise FileExistsError(f"Output CSV is a symlink: {target}")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=output,
            prefix=".results.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def _checkpoint_id(metadata: dict) -> str:
    value = metadata.get("checkpoint_sha256", "")
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def predict(model, pair, mode: str, depth: int | None):
    import torch

    with torch.inference_mode():
        if mode == "fixed":
            if depth not in (1, 2, 3, 4):
                raise ValueError("Fixed mode requires depth 1, 2, 3, or 4")
            density, trace = model.forward_fixed_k(pair, depth, return_trace=True)
            if trace.get("executed_complete_stages") != list(range(1, depth + 1)):
                raise RuntimeError("Fixed Fusion Depth execution trace mismatch")
            selected = depth
        elif mode == "adaptive":
            density, auxiliary = model.forward_dynamic(pair, return_aux=True)
            selected = int(auxiliary["selected_k"][0].item())
            if auxiliary.get("selected_k_counts") != {selected: 1}:
                raise RuntimeError("Instance-Adaptive Fusion Depth execution trace mismatch")
        else:
            raise ValueError(f"Unsupported inference mode: {mode}")
    if selected not in (1, 2, 3, 4):
        raise RuntimeError(f"Invalid Selected Depth returned by model: {selected}")
    return density, selected


def _prediction_count(dataset: str, density) -> float:
    import numpy as np

    array = density[0, 0].detach().float().cpu().numpy()
    expected = tuple(value // 8 for value in DATASET_SIZE_HW[dataset])
    if array.shape != expected or not np.isfinite(array).all():
        raise RuntimeError(f"Invalid predicted density map shape/content: {array.shape}")
    count = (float(array.sum(dtype=np.float64)) if dataset == "DroneRGBT"
             else float(array.sum()))
    if not np.isfinite(count):
        raise RuntimeError("Predicted Count is not finite")
    return count, array


def _resolve_samples(args: argparse.Namespace) -> list[Sample]:
    if args.rgb is not None:
        if args.tir is None:
            raise ValueError("--tir is required when --rgb is used")
        if args.manifest is not None or args.input_dir is not None:
            raise ValueError("Single-pair and batch input options cannot be combined")
        return [single_sample(args.dataset, args.rgb, args.tir, args.gt, args.sample_id)]
    if args.tir is not None or args.gt is not None or args.sample_id is not None:
        raise ValueError("--tir, --gt, and --sample-id require --rgb")
    input_dir = args.input_dir or Path("examples") / args.dataset
    if args.manifest is not None:
        return read_manifest(args.dataset, args.manifest, input_dir)
    return discover_samples(args.dataset, input_dir)


def run(args: argparse.Namespace, *, model_loader=load_model) -> Path:
    if args.mode == "fixed" and args.depth is None:
        raise ValueError("--depth is required for --mode fixed")
    if args.mode == "adaptive" and args.depth is not None:
        raise ValueError("--depth must be omitted for --mode adaptive")
    samples = _resolve_samples(args)
    model, checkpoint = model_loader(args.dataset, args.checkpoints_root, args.device)
    import torch

    device = torch.device(args.device)
    rows = []
    scope = "single" if args.rgb is not None else "batch"
    mode = f"k={args.depth}" if args.mode == "fixed" else "adaptive"
    for index, sample in enumerate(samples):
        pair, original_hw = preprocess_pair(
            args.dataset, sample.rgb, sample.tir, return_original_hw=True
        )
        density, selected = predict(model, pair.to(device), args.mode, args.depth)
        predicted, _ = _prediction_count(args.dataset, density)
        ground_truth = (ground_truth_count(args.dataset, sample.gt, original_hw)
                        if sample.gt is not None else None)
        absolute_error = abs(predicted - ground_truth) if ground_truth is not None else None
        count_error = predicted - ground_truth if ground_truth is not None else None
        relative_error = (absolute_error / ground_truth * 100.0
                          if ground_truth not in (None, 0) else None)
        rows.append({
            "row_type": "sample",
            "dataset": args.dataset,
            "scope": scope,
            "sample_id": sample.sample_id,
            "mode": mode,
            "ground_truth": "" if ground_truth is None else ground_truth,
            "predicted_count": repr(predicted),
            "selected_depth": selected,
            "count_error": "" if count_error is None else repr(count_error),
            "absolute_error": "" if absolute_error is None else repr(absolute_error),
            "relative_error_percent": "" if relative_error is None else repr(relative_error),
            "checkpoint_sha256": _checkpoint_id(checkpoint),
        })
        if len(samples) > 1:
            print(f"Sample ID: {sample.sample_id}")
        print(f"Predicted Count: {predicted:.2f}")
        print(f"Selected Depth: {selected}")
        if ground_truth is not None:
            print(f"Ground Truth: {ground_truth}")
        if index + 1 < len(samples):
            print()
    return _replace_results_csv(args.output_dir or Path("outputs"), args.dataset, rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed Fusion Depth and Instance-Adaptive Fusion Depth inference"
    )
    parser.add_argument("--dataset", choices=("DroneRGBT", "RGBT-CC"), required=True)
    parser.add_argument("--rgb", type=Path, help="single RGB image")
    parser.add_argument("--tir", type=Path, help="single thermal infrared image")
    parser.add_argument("--gt", type=Path, help="optional single XML/JSON point annotation")
    parser.add_argument("--sample-id", help="optional ID for single-pair output")
    parser.add_argument("--input-dir", type=Path,
                        help="batch root (default: examples/<dataset>/)")
    parser.add_argument("--manifest", type=Path,
                        help="optional CSV/JSONL with explicit sample-ID paths")
    parser.add_argument("--mode", choices=("fixed", "adaptive"), required=True)
    parser.add_argument("--depth", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="output root (results are written under <root>/<dataset>/)")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
