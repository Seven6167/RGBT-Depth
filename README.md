# Instance-Adaptive Fusion-Depth Inference for RGB-T Crowd Counting on Consumer Drones

## Overview

RGB-T crowd counting on consumer drones benefits from complementary visible and thermal cues. However, deeper multimodal fusion does not consistently improve counting accuracy across scenes, while using the same fusion depth for every input can incur unnecessary computation.

Our method combines multi-frequency-enhanced feature fusion, self-guided multi-depth joint training, and a **Fusion Depth Controller**. For each RGB-T image pair, the controller selects a fusion depth $k \in \{1,2,3,4\}$; only the selected fusion stages are executed before a shared density prediction head estimates the crowd count.

This repository provides **inference and evaluation** code, frozen checkpoints, and paired examples for **DroneRGBT** and **RGBT-CC**. Training code is not included.

## Installation

Validated on an NVIDIA RTX 4090 with Python 3.10.8, PyTorch 2.1.2, torchvision 0.16.2, CUDA 11.8, `timm` 1.0.28, and `mamba-ssm` 1.2.0.post1.

```bash
conda create -n rgbt-depth python=3.10.8 -y
conda activate rgbt-depth
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

A binary-compatible `mamba-ssm==1.2.0.post1` installation providing `selective_scan_cuda` is also required; its build must match PyTorch, CUDA, and the CXX11 ABI. Verify it with `python -c "import torch, selective_scan_cuda; print(torch.cuda.is_available())"`.

## Checkpoints and Examples

Each dataset has a frozen shared-network checkpoint (three `best.pt.part*` shards), a `depth_controller.pt`, and a `manifest.json` under `checkpoints/<dataset>/`. The loader automatically reconstructs the weights, verifies SHA256 hashes, and strictly loads the model.

The `examples/DroneRGBT/` and `examples/RGBT-CC/` directories each contain **15 RGB/TIR pairs with ground-truth annotations**. These are curated demonstrations selected using crowd counts, adaptive fusion depths, and historical prediction errors—not representative samples for reporting official test-set performance.

## Inference

Run commands from the repository root. Example: adaptive inference on one DroneRGBT pair.

```bash
python -m inference.run \
  --dataset DroneRGBT \
  --rgb examples/DroneRGBT/RGB/85.jpg \
  --tir examples/DroneRGBT/TIR/85R.jpg \
  --gt examples/DroneRGBT/GT/85R.xml \
  --mode adaptive --device cuda:0
```

For a fixed depth, use `--mode fixed --depth K` (`K` = 1, 2, 3, or 4). For batch inference, use `--input-dir examples/<dataset> --manifest examples/<dataset>/manifest.csv` instead of the single-pair image arguments. Ground truth is optional for inference.

## Evaluation

Evaluate all four fixed depths and the adaptive mode on the included examples:

**DroneRGBT**

```bash
python -m evaluation.run \
  --dataset DroneRGBT --dataset-root examples/DroneRGBT \
  --manifest examples/DroneRGBT/manifest.csv \
  --scope examples-15 --mode all --device cuda:0
```

**RGBT-CC**

```bash
python -m evaluation.run \
  --dataset RGBT-CC --dataset-root examples/RGBT-CC \
  --manifest examples/RGBT-CC/manifest.csv \
  --scope examples-15 --mode all --device cuda:0
```

Evaluation reports **MAE, RMSE, and GAME(0–3)**. Add `--measure-efficiency --warmup 20 --repeats 100` to measure GPU latency and FPS. Official full-test evaluation requires the original datasets and audited test-membership records, which are not included here.

## Outputs

Inference and evaluation write `outputs/<dataset>/results.csv` by default. Inference records predicted counts and selected depths; evaluation adds per-image errors and per-mode metric summaries. An all-mode evaluation of 15 pairs produces **75 sample rows and 5 summary rows**. A successful run replaces the previous results CSV for the same dataset.
