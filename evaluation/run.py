"""Fixed-depth and adaptive counting evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from inference.checkpoint import load_model
from inference.data import discover_samples, preprocess_pair, read_manifest
from inference.run import _checkpoint_id, _replace_results_csv, predict

from .efficiency import measure
from .metrics import DENSITY_HW, aggregate, game_per_image, ground_truth_map


EXPECTED_SAMPLES = {
    ("DroneRGBT", "official-test"): 1800,
    ("RGBT-CC", "official-test"): 800,
    ("DroneRGBT", "examples-15"): 15,
    ("RGBT-CC", "examples-15"): 15,
    ("DroneRGBT", "examples-50"): 50,
    ("RGBT-CC", "examples-50"): 50,
}
DRONE_MEMBERSHIP_SHA256 = "092ead003f5735c1f43b825e2b059eb309f8088ff0c5399eab26ff36978615ed"


def _json_digest(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_official_membership(dataset: str, samples, path: Path | None) -> dict:
    if path is None:
        raise ValueError("Official Test requires --official-membership")
    raw = path.read_bytes()
    record = json.loads(raw)
    if dataset == "DroneRGBT":
        if hashlib.sha256(raw).hexdigest() != DRONE_MEMBERSHIP_SHA256:
            raise RuntimeError("DroneRGBT official Test membership hash mismatch")
        pairs = record.get("pairs", [])
        expected = {row["id"]: (row["rgb"], row["tir"], row["gt"]) for row in pairs}
        actual = {sample.gt.stem: (sample.rgb.name, sample.tir.name, sample.gt.name)
                  for sample in samples}
        if (record.get("dataset") != dataset or record.get("split") != "Official Test"
                or record.get("expected_count") != 1800 or len(expected) != 1800
                or actual != expected):
            raise RuntimeError("DroneRGBT official Test membership mismatch")
        return {"manifest_sha256": hashlib.sha256(raw).hexdigest()}
    if record.get("status") != "RGBT_CC_OFFICIAL_SPLITS_AUDITED":
        raise RuntimeError("Unrecognized RGBT-CC official membership manifest")
    tests = record.get("memberships", {}).get("test", [])
    expected_hash = record.get("split_membership_sha256", {}).get("test")
    expected_ids = {row.get("sample_id") for row in tests}
    if (len(tests) != 800 or len(expected_ids) != 800
            or _json_digest(tests) != expected_hash
            or expected_ids != {sample.sample_id for sample in samples}):
        raise RuntimeError("RGBT-CC official Test membership mismatch")
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "test_membership_sha256": expected_hash}


def _samples(args):
    samples = (read_manifest(args.dataset, args.manifest, args.dataset_root)
               if args.manifest else discover_samples(args.dataset, args.dataset_root))
    expected = EXPECTED_SAMPLES[(args.dataset, args.scope)]
    if len(samples) != expected:
        raise RuntimeError(f"{args.scope} requires {expected} samples; found {len(samples)}")
    if args.dataset == "RGBT-CC" and args.scope == "official-test":
        if any(not sample.sample_id.startswith("test:") for sample in samples):
            raise RuntimeError("RGBT-CC official Test manifest IDs must start with 'test:'")
    if any(sample.gt is None for sample in samples):
        raise RuntimeError("Evaluation requires GT for every sample")
    membership = (_verify_official_membership(
        args.dataset, samples, getattr(args, "official_membership", None)
    ) if args.scope == "official-test" else None)
    return samples, membership


def _modes(mode: str, depth: int | None):
    if mode == "all":
        return [("fixed", k) for k in (1, 2, 3, 4)] + [("adaptive", None)]
    return [(mode, depth)]


def evaluate(args) -> Path:
    if args.mode == "fixed" and args.depth is None:
        raise ValueError("--mode fixed requires --depth")
    if args.mode != "fixed" and args.depth is not None:
        raise ValueError("--depth is valid only with --mode fixed")
    if not args.measure_efficiency and (args.flops_profile or args.controller_flops_g is not None):
        raise ValueError("FLOPs options require --measure-efficiency")
    samples, membership = _samples(args)
    model, checkpoint = load_model(args.dataset, args.checkpoints_root, args.device)
    import torch

    device = torch.device(args.device)
    summaries, groups = [], []
    for mode, depth in _modes(args.mode, args.depth):
        rows = []
        for sample in samples:
            pair, original_hw = preprocess_pair(
                args.dataset, sample.rgb, sample.tir, return_original_hw=True
            )
            prediction, chosen = predict(model, pair.to(device), mode, depth)
            density = prediction[0, 0].detach().float().cpu().numpy()
            if density.shape != DENSITY_HW[args.dataset] or not np.isfinite(density).all():
                raise RuntimeError(f"Invalid density prediction: {sample.sample_id}")
            target = ground_truth_map(args.dataset, sample.gt, sample.rgb, original_hw)
            if args.dataset == "DroneRGBT":
                predicted_count = float(prediction.float().sum().item())
                ground_truth = float(target.sum(dtype=np.float32))
            else:
                predicted_count = float(density.astype(np.float64).sum())
                ground_truth = float(target.astype(np.float64).sum())
            game = game_per_image(density, target)
            count_error = predicted_count - ground_truth
            absolute_error = abs(count_error)
            row = {
                "sample_id": sample.sample_id,
                "mode": f"k={depth}" if mode == "fixed" else "adaptive",
                "selected_depth": chosen,
                "predicted_count": predicted_count,
                "ground_truth": ground_truth,
                "count_error": count_error,
                "absolute_error": absolute_error,
                "relative_error_percent": (absolute_error / ground_truth * 100.0
                                           if ground_truth != 0 else ""),
                "game": game,
            }
            rows.append(row)
        summary = aggregate(rows, args.dataset)
        summary.update({
            "mode": f"k={depth}" if mode == "fixed" else "adaptive",
            "params_m": sum(parameter.numel() for parameter in model.parameters()) / 1e6,
        })
        if args.measure_efficiency:
            efficiency_args = argparse.Namespace(
                dataset=args.dataset, dataset_root=args.dataset_root, manifest=args.manifest,
                mode=mode, depth=depth, checkpoints_root=args.checkpoints_root,
                device=args.device, warmup=args.warmup, repeats=args.repeats,
                operator_flops=False, flops_profile=args.flops_profile,
                controller_flops_g=args.controller_flops_g, output=None,
            )
            measured = measure(efficiency_args, model=model, samples=samples,
                               print_result=False)
            summary.update({
                "flops_g": measured.get("sourced_FLOPs_G", ""),
                "latency_ms": measured["latency_ms"],
                "fps": measured["fps"],
            })
        summaries.append(summary)
        groups.append((rows, summary))

    csv_rows = []
    for rows, summary in groups:
        for row in rows:
            values = {key: value for key, value in row.items() if key != "game"}
            values.update({
                "row_type": "sample", "dataset": args.dataset, "scope": args.scope,
                **{f"GAME({i})": row["game"][i] for i in range(4)},
            })
            csv_rows.append(values)
        csv_rows.append({
            "row_type": "summary", "dataset": args.dataset, "scope": args.scope,
            "mode": summary["mode"], "num_samples": summary["num_images"],
            "MAE": summary["MAE"], "RMSE": summary["RMSE"],
            **{f"GAME({i})": summary[f"GAME({i})"] for i in range(4)},
            "params_m": summary["params_m"], "flops_g": summary.get("flops_g", ""),
            "latency_ms": summary.get("latency_ms", ""), "fps": summary.get("fps", ""),
            "checkpoint_sha256": _checkpoint_id(checkpoint),
            "membership_sha256": (membership or {}).get("manifest_sha256", ""),
            "test_membership_sha256":
                (membership or {}).get("test_membership_sha256", ""),
        })
    output = _replace_results_csv(args.output_dir, args.dataset, csv_rows)
    for item in summaries:
        values = " ".join(f"GAME({i})={item[f'GAME({i})']:.4f}" for i in range(4))
        print(f"{item['mode']}: N={item['num_images']} MAE={item['MAE']:.4f} "
              f"RMSE={item['RMSE']:.4f} {values}")
    return output


def build_parser():
    parser = argparse.ArgumentParser(description="RGB-T crowd-counting evaluation")
    parser.add_argument("--dataset", choices=("DroneRGBT", "RGBT-CC"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--official-membership", type=Path)
    parser.add_argument("--scope", choices=("official-test", "examples-15", "examples-50"), required=True)
    parser.add_argument("--mode", choices=("all", "fixed", "adaptive"), default="all")
    parser.add_argument("--depth", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--measure-efficiency", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--flops-profile", type=Path)
    parser.add_argument("--controller-flops-g", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"),
                        help="output root (results are written under <root>/<dataset>/)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        evaluate(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
