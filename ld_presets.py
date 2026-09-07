"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus
dem OR-Demo-Portfolio, siehe sp_presets.py in spectral-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import ld_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _shape_caster(v):
    return v if v in C.SHAPES else C.DEFAULT_SHAPE


def _quality_function_caster(v):
    return v if v in C.QUALITY_FUNCTIONS else C.DEFAULT_QUALITY_FUNCTION


SETTING_SPECS = {
    "n_points_slider": SettingSpec("n", int, C.DEFAULT_N_POINTS, C.N_POINTS_MIN, C.N_POINTS_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "spread_slider": SettingSpec("spread", float, C.DEFAULT_SPREAD, C.SPREAD_MIN, C.SPREAD_MAX),
    "density_imbalance_slider": SettingSpec(
        "dimb", float, C.DEFAULT_DENSITY_IMBALANCE, C.DENSITY_IMBALANCE_MIN, C.DENSITY_IMBALANCE_MAX
    ),
    "bridge_strength_slider": SettingSpec(
        "bridge", float, C.DEFAULT_BRIDGE_STRENGTH, C.BRIDGE_STRENGTH_MIN, C.BRIDGE_STRENGTH_MAX
    ),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "shape_radio": SettingSpec("shape", _shape_caster, C.DEFAULT_SHAPE),
    "n_neighbors_slider": SettingSpec(
        "nn", int, C.DEFAULT_N_NEIGHBORS, C.N_NEIGHBORS_MIN, C.N_NEIGHBORS_MAX
    ),
    "quality_function_radio": SettingSpec("qf", _quality_function_caster, C.DEFAULT_QUALITY_FUNCTION),
    "resolution_slider": SettingSpec(
        "res", float, C.DEFAULT_RESOLUTION, C.RESOLUTION_MIN, C.RESOLUTION_MAX
    ),
    "cpm_resolution_slider": SettingSpec(
        "cpmres", float, C.DEFAULT_CPM_RESOLUTION, C.CPM_RESOLUTION_MIN, C.CPM_RESOLUTION_MAX
    ),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(
    n_points, k, spread, density_imbalance, bridge_strength, seed, shape, n_neighbors, quality_function, resolution
):
    try:
        st.query_params["n"] = str(int(n_points))
        st.query_params["k"] = str(int(k))
        st.query_params["spread"] = str(spread)
        st.query_params["dimb"] = str(density_imbalance)
        st.query_params["bridge"] = str(bridge_strength)
        st.query_params["seed"] = str(int(seed))
        st.query_params["shape"] = shape
        st.query_params["nn"] = str(int(n_neighbors))
        st.query_params["qf"] = quality_function
        if quality_function == "cpm":
            st.query_params["cpmres"] = str(resolution)
        else:
            st.query_params["res"] = str(resolution)
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_points_slider"] = p["n_points"]
    st.session_state["k_slider"] = p["k"]
    st.session_state["spread_slider"] = p["spread"]
    st.session_state["density_imbalance_slider"] = p["density_imbalance"]
    st.session_state["bridge_strength_slider"] = p["bridge_strength"]
    st.session_state["shape_radio"] = p["shape"]
    st.session_state["n_neighbors_slider"] = p["n_neighbors"]
    st.session_state["quality_function_radio"] = p["quality_function"]
    if p["quality_function"] == "cpm":
        st.session_state["cpm_resolution_slider"] = p["resolution"]
    else:
        st.session_state["resolution_slider"] = p["resolution"]
    st.session_state["seed_input"] = p["seed"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
