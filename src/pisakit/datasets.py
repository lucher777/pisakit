"""Bundled real OECD PISA 2022 micro-slice and 15-system benchmark loaders."""
from __future__ import annotations

from importlib.resources import files

import numpy as np
import pandas as pd


def _data_root():
    """Return the package resource traversable for bundled PISA 2022 data files."""
    return files("pisakit").joinpath("data")


def load_pisa2022_micro_slice() -> pd.DataFrame:
    """Load the bundled 960-student de-identified OECD PISA 2022 micro-slice.

    Contains 60 schools (15 schools per education system across ``FIN``, ``JPN``,
    ``KOR``, and ``EST``; 16 complete students per school) with five mathematics
    plausible values (``PV1MATH``-``PV5MATH``), ``escs``, ``female``,
    ``hi_escs``, final student weight ``W_FSTUWT``, and all 80 official Fay BRR
    replicate weights (``W_FSTURWT1``-``W_FSTURWT80``).

    Returns:
        DataFrame with ``960`` rows and ``91`` columns cast to ``float64`` for
        exact numerical computations.
    """
    npz_res = _data_root().joinpath("pisa2022_4cnt_slice.npz")
    with npz_res.open("rb") as fh:
        raw = np.load(fh, allow_pickle=False)
        data: dict[str, object] = {
            "CNT": raw["cnt"].astype(str),
            "CNTSCHID": raw["cntschid"].astype(np.int64),
            "STRATUM": raw["stratum"].astype(np.int64),
            "escs": raw["escs"].astype(np.float64),
            "female": raw["female"].astype(np.float64),
            "hi_escs": raw["hi_escs"].astype(np.float64),
        }
        pvs = raw["pvs"].astype(np.float64)
        for idx in range(pvs.shape[1]):
            data[f"PV{idx + 1}MATH"] = pvs[:, idx]
        data["W_FSTUWT"] = raw["w_final"].astype(np.float64)
        reps = raw["w_reps"].astype(np.float64)
        for idx in range(reps.shape[1]):
            data[f"W_FSTURWT{idx + 1}"] = reps[:, idx]
    return pd.DataFrame(data)


def load_pisa2022_system_summary() -> pd.DataFrame:
    """Load full-sample PISA 2022 benchmark metrics across 15 education systems.

    Computed across full country samples using all ``M=10`` mathematics
    plausible values and ``R=80`` Fay BRR weights, reporting sample counts,
    mean scores, overall ESCS gradients, Mundlak within- and between-school
    slopes, and Cinelli-Hazlett robustness values (``rv_q`` and ``rv_qa``).

    Returns:
        DataFrame with ``15`` rows sorted in bundled benchmark order.
    """
    csv_res = _data_root().joinpath("pisa2022_15cnt_summary.csv")
    with csv_res.open("r", encoding="utf-8") as fh:
        return pd.read_csv(fh)
