"""Shared pytest fixtures for deterministic synthetic LSA datasets."""

from __future__ import annotations

import pandas as pd
import pytest

from pisakit import DesignSpec, simulate_lsa_survey


@pytest.fixture(scope="session")
def single_system_spec() -> DesignSpec:
    """Compact PISA-like design with 5 PVs and 40 Fay BRR replicates."""
    return DesignSpec.from_preset("pisa", n_reps=40, n_pvs=5)


@pytest.fixture(scope="session")
def single_system_df() -> pd.DataFrame:
    """Deterministic single-system synthetic dataset with known true slopes."""
    return simulate_lsa_survey(
        n_systems=1,
        n_strata_per_system=20,
        students_per_school=16,
        n_pvs=5,
        n_reps=40,
        pv_sd=22.0,
        school_icc=0.30,
        true_escs=30.0,
        true_female=-8.0,
        true_treatment=20.0,
        true_school_escs=25.0,
        system_heterogeneity_sd=0.0,
        seed=1234,
    )


@pytest.fixture(scope="session")
def multi_system_spec() -> DesignSpec:
    """Multi-system PISA-like design preset for cross-national tests."""
    return DesignSpec.from_preset("pisa", n_reps=40, n_pvs=5)


@pytest.fixture(scope="session")
def multi_system_df() -> pd.DataFrame:
    """Deterministic 4-system synthetic dataset with slope heterogeneity."""
    return simulate_lsa_survey(
        n_systems=4,
        n_strata_per_system=10,
        students_per_school=15,
        n_pvs=5,
        n_reps=40,
        pv_sd=24.0,
        school_icc=0.25,
        true_escs=28.0,
        true_female=-6.0,
        true_treatment=18.0,
        true_school_escs=15.0,
        system_heterogeneity_sd=9.0,
        seed=4321,
    )
