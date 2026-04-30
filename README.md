# PPO from Scratch

A clean implementation of **Proximal Policy Optimization (PPO)** in PyTorch, built from scratch.

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd ppo
```

### 2. Install dependencies

Pick the command that matches your hardware:

**CPU** (works everywhere):
```bash
uv sync --extra cpu
```

**GPU** (NVIDIA, CUDA 12.4, driver ≥ 525):
```bash
uv sync --extra cuda
```

| CUDA version | Command |
|---|---|
| 12.4 | `uv sync --extra cuda` (default) |
| 12.1 / 11.8 | Change `cu124` → `cu121` / `cu118` in `pyproject.toml` |

> [!TIP]
> Not sure which CUDA version you have? Run `nvidia-smi` — the version appears in the top-right corner.

### 3. Run

```bash
uv run python main.py
```

Or activate the virtual environment directly:

```bash
source .venv/bin/activate
python main.py
```

## Project Structure

```
ppo/
├── main.py          # Entry point
├── pyproject.toml   # Project metadata & dependencies
└── README.md
```

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural network & tensor operations |
| `torchvision` | Vision utilities |
| `numpy` | Numerical computing |
| `gymnasium` | RL environments (Classic Control) |
