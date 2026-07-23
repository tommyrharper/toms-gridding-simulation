import _repo_path  # noqa: F401  — put src/ on path before package imports
from demo_config import DEMO
from demo_core import run_demo

from gridding_sim.arrays import list_arrays
from gridding_sim.plotting import plot_demo_summary


def main() -> None:
    cfg = DEMO

    print(len(list_arrays()), "configurations available")
    result = run_demo(cfg)

    plot_demo_summary(
        result.u,
        result.v,
        cfg.radio_array,
        result.beam,
        result.beam_cell,
        result.img_dft,
        result.d_sph,
        result.d_lm,
        result.vmax,
        show_plot=cfg.show_plot,
    )


if __name__ == "__main__":
    main()
