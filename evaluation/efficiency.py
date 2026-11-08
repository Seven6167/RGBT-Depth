"""Parameters, sourced FLOPs, and batch-1 forward latency."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from pathlib import Path

from inference.checkpoint import load_model
from inference.data import discover_samples, preprocess_pair, read_manifest


def _flops_profile(path: Path) -> tuple[dict[int, float], float | None]:
    if path.suffix.lower() == ".json":
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("unit") != "FLOPs":
            raise ValueError("JSON FLOPs profile must use unit=FLOPs")
        values = {int(k): float(v) / 1e9
                  for k, v in record["fixed_path_flops"].items()}
        controller = record.get("controller_flops")
        controller = None if controller is None else float(controller) / 1e9
    else:
        with path.open(encoding="utf-8", newline="") as stream:
            values = {int(row["k"]): float(row["flops_g"])
                      for row in csv.DictReader(stream)}
        controller = None
    if set(values) != {1, 2, 3, 4} or any(value <= 0 for value in values.values()):
        raise ValueError("FLOPs profile must contain positive k=1,2,3,4 values")
    return values, controller


def _operator_profile(model, image, mode, depth):
    import torch
    try:
        from thop import profile
    except ImportError as error:
        raise RuntimeError("THOP is required only for --operator-flops") from error

    class ActivePath(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, value):
            if mode == "fixed":
                return self.model.forward_fixed_k(value, depth)
            return self.model.forward_dynamic(value)

    with torch.inference_mode():
        macs, _ = profile(ActivePath().eval(), inputs=(image,), verbose=False)
    if float(macs) <= 0:
        raise RuntimeError("THOP returned an invalid MAC count")
    return float(macs)


def measure(args, *, model=None, samples=None, print_result: bool = True) -> dict:
    if args.mode == "fixed" and args.depth not in (1, 2, 3, 4):
        raise ValueError("--mode fixed requires --depth")
    if args.mode == "adaptive" and args.depth is not None:
        raise ValueError("--mode adaptive does not accept --depth")
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("--warmup must be nonnegative and --repeats must be positive")
    if samples is None:
        samples = (read_manifest(args.dataset, args.manifest, args.dataset_root)
                   if args.manifest else discover_samples(args.dataset, args.dataset_root))
    if model is None:
        model, _ = load_model(args.dataset, args.checkpoints_root, args.device)
    import torch

    device = torch.device(args.device)
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    times, selected, operator_macs = [], [], []
    for sample in samples:
        image = preprocess_pair(args.dataset, sample.rgb, sample.tir).to(device)

        def forward():
            if args.mode == "fixed":
                return model.forward_fixed_k(image, args.depth)
            return model.forward_dynamic(image)

        with torch.inference_mode():
            for _ in range(args.warmup):
                forward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            for _ in range(args.repeats):
                if device.type == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    forward()
                    end.record()
                    end.synchronize()
                    elapsed = start.elapsed_time(end)
                else:
                    start_time = time.perf_counter()
                    forward()
                    elapsed = (time.perf_counter() - start_time) * 1000.0
                times.append(float(elapsed))
            if args.mode == "adaptive":
                _, auxiliary = model.forward_dynamic(image, return_aux=True)
                selected.append(int(auxiliary["selected_k"][0].item()))
            else:
                selected.append(args.depth)
        if args.operator_flops:
            operator_macs.append(_operator_profile(model, image, args.mode, args.depth))

    mean_ms = statistics.mean(times)
    result = {
        "dataset": args.dataset,
        "mode": args.mode,
        "depth": args.depth,
        "num_images": len(samples),
        "batch_size": 1,
        "warmup_per_image": args.warmup,
        "repeats_per_image": args.repeats,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "latency_ms": mean_ms,
        "fps": 1000.0 / mean_ms,
        "selected_depths": selected,
        "latency_scope": "model forward only; adaptive includes the controller",
    }
    if args.operator_flops:
        macs = statistics.mean(operator_macs)
        result.update({
            "operator_profile_MACs_G": macs / 1e9,
            "operator_profile_FLOPs_G": 2.0 * macs / 1e9,
            "operator_profile_convention": "1 MAC = 2 FLOPs",
            "operator_profile_complete": False,
            "operator_profile_limitation":
                "THOP does not fully count custom Mamba/selective-scan CUDA operators",
        })
    if args.flops_profile:
        values, profile_controller = _flops_profile(args.flops_profile)
        controller = (args.controller_flops_g
                      if args.controller_flops_g is not None else profile_controller)
        if args.mode != "adaptive" or controller is not None:
            active = statistics.mean(values[k] for k in selected)
            overhead = controller if args.mode == "adaptive" else 0.0
            result.update({
                "sourced_FLOPs_G": active + overhead,
                "sourced_FLOPs_include_controller": args.mode == "adaptive",
                "flops_profile_sha256":
                    hashlib.sha256(args.flops_profile.read_bytes()).hexdigest(),
                "flops_scope": "verified active-path table; not recomputed by this run",
            })
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if print_result:
        printable = {key: value for key, value in result.items() if key != "selected_depths"}
        print(json.dumps(printable, indent=2))
    return result


def build_parser():
    parser = argparse.ArgumentParser(description="RGB-T inference efficiency")
    parser.add_argument("--dataset", choices=("DroneRGBT", "RGBT-CC"), required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--mode", choices=("fixed", "adaptive"), required=True)
    parser.add_argument("--depth", type=int, choices=(1, 2, 3, 4))
    parser.add_argument("--checkpoints-root", type=Path, default=Path("checkpoints"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--operator-flops", action="store_true")
    parser.add_argument("--flops-profile", type=Path)
    parser.add_argument("--controller-flops-g", type=float)
    parser.add_argument("--output", type=Path,
                        help="optional JSON output; results print to the terminal by default")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        measure(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
