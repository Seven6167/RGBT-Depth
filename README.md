# Instance-Adaptive Fusion-Depth Inference for RGB-T Crowd Counting on Consumer Drones

Official inference and evaluation code for our IEEE Transactions on Consumer Electronics submission. The method selects the fusion depth for each RGB-T image pair using a Fusion Depth Controller, with four available depths ($k=1,2,3,4$).

## Installation

Tested environment: Python 3.10.8, PyTorch 2.1.2 + CUDA 11.8, torchvision 0.16.2, `timm` 1.0.28, `mamba-ssm` 1.2.0.post1, and NVIDIA RTX 4090.

```bash
conda create -n rgbt-tce python=3.10.8 -y
conda activate rgbt-tce
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Install a binary-compatible `mamba-ssm==1.2.0.post1` build that provides `selective_scan_cuda` (PyTorch 2.1, CUDA 11.8, matching CXX11 ABI). If using the previously validated GPU environment, retain its working PyTorch and Mamba installation.

## Checkpoints

```text
checkpoints/
├── DroneRGBT/
│   ├── manifest.json
│   ├── depth_controller.pt
│   └── shared_network/best.pt.part001 ... best.pt.part007
└── RGBT-CC/
    ├── manifest.json
    ├── depth_controller.pt
    └── shared_network/best.pt.part001 ... best.pt.part007
```

To keep each checkpoint file under 25 MB, the shared-network weights are split into seven shards per dataset. The loader automatically reconstructs and verifies the weights (including SHA256 checks) before strict loading; no manual merging is needed.

## Evaluation

Run the following commands from the repository root. Each evaluates all four fixed fusion depths and instance-adaptive depth on the **50 paired examples** provided for that dataset.

**DroneRGBT**

```bash
python -m evaluation.run --dataset DroneRGBT \
  --dataset-root examples/DroneRGBT \
  --manifest examples/DroneRGBT/manifest.csv \
  --scope examples-50 --mode all --device cuda:0
```

**RGBT-CC**

```bash
python -m evaluation.run --dataset RGBT-CC \
  --dataset-root examples/RGBT-CC \
  --scope examples-50 --mode all --device cuda:0
```

For one mode, use `--mode adaptive` or `--mode fixed --depth K` (where `K` is 1–4). To measure GPU latency and FPS, add `--measure-efficiency --warmup 20 --repeats 100`.

To limit package size, only **50 paired examples per dataset** are included instead of the complete official test sets. These subsets are for code verification and demonstration; their metrics must not be reported as full-test results. Full test evaluation requires the original datasets and verified test membership files.

## Outputs

The `outputs/` directory is created automatically after evaluation:

```text
outputs/
├── DroneRGBT/results.csv
└── RGBT-CC/results.csv
```

Each CSV records **per-sample predictions** (sample ID, mode, selected depth, predicted/ground-truth counts, and count errors) and **per-mode summaries** (MAE, RMSE, and GAME(0–3)). With `--mode all` and 50 pairs, each CSV contains 250 sample rows and five summary rows. Optional efficiency measurements are included only when requested.

Rerunning evaluation replaces only the corresponding dataset's `results.csv`; the other dataset's results remain unchanged.
