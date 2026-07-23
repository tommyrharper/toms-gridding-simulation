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
| `scripts/` | Runnable demos / CLIs (`demo_observe.py`, `demo_app.py` Streamlit UI, shared pipeline in `demo_core.py`, `demo_ml_step.py`) |
| `ml/` | PyTorch MLP baseline: raw padded (u,v,V) visibilities -> flattened dirty image (foundation only — no training loop yet) |
| `data/` | Generated datasets (gitignored) |
| `tests/` | Pytest suite |

**Later (not created yet):** `metrics/`, `benchmarks/`, `runs/`.

## Run the demo

```sh
uv sync
uv run scripts/demo_observe.py
```

If you see `ModuleNotFoundError: No module named 'gridding_sim'`, the demo
script bootstraps `src/` automatically via `scripts/_repo_path.py`. Re-run
`uv sync` once, then try the command again.

## Interactive demo

`scripts/demo_app.py` is a Streamlit front-end for `demo_observe.py`: sliders and
buttons for every `DemoConfig` field, with a Generate button that runs the same
observe -> DFT/FFT dirty-image pipeline and shows the resulting plots and stats.

```sh
uv sync --group demo
uv run streamlit run scripts/demo_app.py
```

Generation isn't instant — FFT gridding is a pure-Python loop over every
visibility, so large arrays / long durations / large npix can take up to a
minute. The controls are batched behind the Generate button rather than
recomputing on every slider tick.

## Testing

```sh
uv run pytest
```

## Notes

Check if there is anything where the least-misfit function performs worst.
Modify my code to cut out the dodgy bits at the side of the image after gridding +iFFT.
Choose the cutoff.

Plug in different gridding functions and see the difference.

We need a metric.

- I need to understand the physics better
- Identify what the leading problem is
  - Reading the current techniques
  - What are people worried about
  - Where are they spending their compute
  - Concern about accuracy vs compute (what do they care about more)

People wouldn't get the paper published if they used ML methods.

What are the most popular tools:
1. Get a list of the most popular tools
2. Decide what metrics most important for those methods

For next meeting - do time sensitive literature review:
- last 5 years
- what techniques are most popular
- what pipelines do people trust
  - what is the algorithm
  - what is the codebase
  - is it open source
  - do they have access to it
  - is there some part of the pipeline that isn't optimised
    - why are they not using the optimised version?
      - people aren't using them, not being cited
      - is the whole community working on inertia

Viable paper could be:
- Here is a benchmarking system for all the methods
- critique of current ml methods => validation
