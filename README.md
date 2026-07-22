# toms-gridding-simulation

Analytical radio interferometry simulation for gridding and dirty-image work (UROP).

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and `pytest` for testing.

```sh
uv sync
```

## Repo layout

| Path | Role |
|---|---|
| `src/gridding_sim/` | Library: arrays, observe, simulate (DFT ground truth), imaging (grid+FFT), gridtools (kernels), diagnostics, plotting |
| `configs/` | Antenna array `.cfg` files |
| `scripts/` | Runnable demos / CLIs (`demo_observe.py`) |
| `data/` | Generated datasets (gitignored) |
| `tests/` | Pytest suite |

**Later (not created yet):** `metrics/`, `ml/`, `benchmarks/`, `runs/`.

## Run the demo

```sh
uv sync
uv run scripts/demo_observe.py
```

If you see `ModuleNotFoundError: No module named 'gridding_sim'`, the demo
script bootstraps `src/` automatically via `scripts/_repo_path.py`. Re-run
`uv sync` once, then try the command again.

## Testing

```sh
uv run pytest
```
