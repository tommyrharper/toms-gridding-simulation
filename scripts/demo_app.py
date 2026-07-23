"""Interactive Streamlit front-end for demo_observe.py.

Sliders and buttons drive a DemoConfig, run_demo() from demo_core.py does the
actual work, and plot_demo_summary() draws the same figure the CLI produces.

    uv sync --group demo
    uv run streamlit run scripts/demo_app.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import _repo_path  # noqa: F401  — put src/ on path before package imports
from demo_config import DEMO, RADIO, DemoConfig
from demo_core import DemoResult, run_demo

from gridding_sim.arrays import list_arrays
from gridding_sim.plotting import plot_demo_summary

st.set_page_config(page_title="Gridding demo", layout="wide")


@st.cache_data(show_spinner=False)
def _run_demo_cached(cfg: DemoConfig) -> DemoResult:
    return run_demo(cfg, verbose=False)


if "rng_seed" not in st.session_state:
    st.session_state.rng_seed = DEMO.rng_seed
if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.cfg = None
    st.session_state.error = None

st.title("Gridding / dirty-image demo")
st.caption(
    "Tune the same knobs as `scripts/demo_config.py` (`DemoConfig`), then hit "
    "**Generate** to run the observe -> DFT/FFT dirty-image pipeline from "
    "`demo_observe.py`."
)

with st.sidebar:
    st.header("Antenna array")
    show_all = st.checkbox(f"Show full catalogue ({len(list_arrays())} arrays)")
    array_options = list_arrays() if show_all else RADIO
    default_array = DEMO.radio_array if DEMO.radio_array in array_options else array_options[0]

    st.header("Sky model")
    # Kept outside the form so the conditional widgets below (flux / n_sources /
    # manual-source table) update immediately when the mode changes, instead of
    # only appearing after the next Generate click.
    sky_mode = st.radio(
        "sky_mode", ["single", "random", "manual"],
        index=["single", "random", "manual"].index(DEMO.sky_mode),
        horizontal=True,
    )

    with st.form("controls"):
        radio_array = st.selectbox(
            "Array", array_options, index=array_options.index(default_array)
        )

        st.header("Pointing & time")
        ra_deg = st.slider("Right ascension [deg]", -180.0, 180.0, DEMO.ra_deg, step=1.0)
        dec_deg = st.slider("Declination [deg]", -90.0, 90.0, DEMO.dec_deg, step=1.0)
        duration_h = st.slider(
            "Duration [hours]", 0.05, 12.0, DEMO.duration_h, step=0.05,
            help="More hours -> more visibilities -> slower FFT gridding.",
        )

        st.header("Imaging grid")
        npix = st.select_slider("npix", options=[64, 128, 256, 512], value=DEMO.npix)
        cell = st.slider("cell [arcsec/pixel]", 0.01, 2.0, DEMO.cell, step=0.01)

        flux = DEMO.flux
        n_sources = DEMO.n_sources
        manual_sources: list[tuple[float, float, float]] = list(DEMO.manual_sources)

        if sky_mode in ("single", "random"):
            flux = st.slider("flux [Jy]", 0.1, 10.0, DEMO.flux, step=0.1)
        if sky_mode == "random":
            n_sources = st.slider("n_sources", 1, 50, DEMO.n_sources)
            st.number_input("rng seed", key="rng_seed", step=1)
        if sky_mode == "manual":
            st.caption("(x_arcsec, y_arcsec, flux_Jy) per row")
            df = pd.DataFrame(
                DEMO.manual_sources, columns=["x_arcsec", "y_arcsec", "flux_Jy"]
            )
            edited = st.data_editor(df, num_rows="dynamic", key="manual_df")
            manual_sources = [tuple(row) for row in edited.to_numpy(dtype=float)]

        submitted = st.form_submit_button("Generate", use_container_width=True)

    if sky_mode == "random":
        if st.button("New random seed", use_container_width=True):
            st.session_state.rng_seed = int(np.random.randint(0, 1_000_000))
            st.rerun()

if submitted:
    cfg = DemoConfig(
        radio_array=radio_array,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        duration_h=duration_h,
        npix=npix,
        cell=cell,
        sky_mode=sky_mode,
        n_sources=n_sources,
        flux=flux,
        manual_sources=manual_sources,
        rng_seed=int(st.session_state.rng_seed),
        show_plot=False,
    )
    st.session_state.cfg = cfg
    st.session_state.error = None
    try:
        with st.spinner("Observing + gridding — the FFT step can take up to a minute for large arrays/durations..."):
            st.session_state.result = _run_demo_cached(cfg)
    except (ValueError, NotImplementedError) as e:
        st.session_state.result = None
        st.session_state.error = str(e)

if st.session_state.error:
    st.error(st.session_state.error)
elif st.session_state.result is None:
    st.info("Set your parameters in the sidebar and click **Generate**.")
else:
    result = st.session_state.result
    cfg = st.session_state.cfg

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Visibilities", f"{result.info['n_vis']:,}")
    c2.metric("Max elevation", f"{result.info['max_elev_deg']:.1f} deg")
    c3.metric("Sources", len(result.sources))
    c4.metric("w-term phase error", f"{result.dphi:.2e} rad")
    st.caption(f"w-term: {result.narrow_field_msg}")

    stat_cols = st.columns(len(result.residual_stats))
    for col, (name, s) in zip(stat_cols, result.residual_stats.items()):
        col.metric(f"{name} inner-field error", f"max {s['max']:.2e}", f"rms {s['rms']:.2e}")

    fig = plot_demo_summary(
        result.u,
        result.v,
        cfg.radio_array,
        result.beam,
        result.beam_cell,
        result.img_dft,
        result.d_sph,
        result.d_lm,
        result.vmax,
        show_plot=False,
    )
    st.pyplot(fig)
    plt.close(fig)
