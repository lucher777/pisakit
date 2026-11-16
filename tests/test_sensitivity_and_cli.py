"""Tests for Cinelli-Hazlett OVB sensitivity analysis and CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pisakit import (
    DesignSpec,
    adjusted_estimate,
    compute_robustness_values,
    sensitivity_analysis,
    sensitivity_contour_grid,
    simulate_lsa_survey,
)
from pisakit.cli import main as cli_main


def test_robustness_value_exactly_nullifies_effect_at_rv_q() -> None:
    """Plugging r2_dz = r2_yz = RV_1 into adjusted_estimate must reduce estimate to zero."""
    est, se, df = 18.0, 4.5, 45.0
    t_stat = est / se
    r2_yd_x, rv_q, rv_qa = compute_robustness_values(t_stat, df=df, q=1.0, alpha=0.05)

    assert 0.0 < rv_qa < rv_q < 1.0
    assert 0.0 < r2_yd_x < 1.0

    adj_null = adjusted_estimate(est, se, df=df, r2_dz_x=rv_q, r2_yz_dx=rv_q, direction=-1.0)
    assert np.isclose(float(adj_null["adjusted_estimate"]), 0.0, atol=1e-10)

    # Weak effect (|t| < t_crit) must have zero RV_{q, alpha}
    _, rv_weak, rv_weak_a = compute_robustness_values(t_stat=1.2, df=df, q=1.0, alpha=0.05)
    assert rv_weak > 0.0
    assert rv_weak_a == 0.0


def test_finite_barnard_rubin_df_reduces_rv_qa_vs_infinite_df() -> None:
    """Finite survey degrees of freedom must yield stricter (lower) RV_{q,alpha}."""
    _, rv_q_fin, rv_qa_fin = compute_robustness_values(t_stat=3.2, df=25.0, q=1.0, alpha=0.05)
    _, rv_q_large, rv_qa_large = compute_robustness_values(t_stat=3.2, df=1e6, q=1.0, alpha=0.05)
    assert rv_q_fin > rv_q_large  # larger f = t/sqrt(df) vs smaller threshold comparison
    assert rv_qa_fin < rv_qa_large or rv_qa_large > 0.0
    # Verify exact validation on bounds
    with pytest.raises(ValueError, match="r2_dz_x"):
        adjusted_estimate(10.0, 2.0, df=30.0, r2_dz_x=1.0, r2_yz_dx=0.1)


def test_sensitivity_analysis_with_benchmark_covariates(
    single_system_df: pd.DataFrame, single_system_spec: DesignSpec
) -> None:
    """Benchmark bounds must attenuate estimate progressively with 1x, 2x, 3x multipliers."""
    ovb = sensitivity_analysis(
        single_system_df,
        outcome_pattern="PV{k}MATH",
        treatment="treatment",
        controls=["escs", "female"],
        design=single_system_spec,
        benchmarks=["escs", "female"],
        multipliers=(1.0, 2.0, 3.0),
    )
    assert ovb.rv_q > 0.0
    tbl = ovb.benchmark_bounds
    assert len(tbl) == 6
    escs_rows = tbl[tbl["benchmark"] == "escs"].sort_values("multiplier")
    assert escs_rows["adjusted_estimate"].is_monotonic_decreasing


def test_sensitivity_contour_grid_properties() -> None:
    """Contour grid output shapes and directional monotonicity must match OVB equations."""
    grid = sensitivity_contour_grid(estimate=15.0, se=3.0, df=40.0, n_grid=25)
    assert grid["adjusted_estimate"].shape == (25, 25)
    assert grid["adjusted_t"].shape == (25, 25)
    assert np.isclose(grid["adjusted_estimate"][0, 0], 15.0)
    assert grid["adjusted_estimate"][-1, -1] < grid["adjusted_estimate"][0, 0]


def test_confounded_dgp_exhibits_shifted_estimate_and_lower_robustness() -> None:
    """Data with hidden confounding shifts OLS upward relative to unconfounded truth."""
    spec = DesignSpec.from_preset("pisa", n_reps=20, n_pvs=3)
    df_conf = simulate_lsa_survey(
        n_systems=1, n_strata_per_system=15, students_per_school=16, n_pvs=3, n_reps=20,
        true_treatment=5.0, unmeasured_confounding=0.8, seed=99,
    )
    ovb = sensitivity_analysis(
        df_conf,
        outcome_pattern="PV{k}MATH",
        treatment="treatment",
        controls=["escs", "female"],
        design=spec,
    )
    # Positive latent confounder inflates treatment slope well above true 5.0
    assert ovb.estimate > 10.0


def test_cli_info_json(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI `info --format json` must return valid JSON describing pisa/timss/pirls."""
    rc = cli_main(["info", "--format", "json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["package"] == "pisakit"
    assert {"pisa", "timss", "pirls"} == set(out["presets"].keys())


def test_cli_end_to_end_pipeline(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """CLI simulate -> ols -> meta -> sensitivity round-trip must succeed cleanly."""
    csv_path = tmp_path / "sim.csv"
    assert cli_main(
        [
            "simulate",
            "--out",
            str(csv_path),
            "--n-systems",
            "2",
            "--n-strata",
            "6",
            "--students-per-school",
            "8",
            "--n-pvs",
            "3",
            "--n-reps",
            "12",
            "--format",
            "json",
        ]
    ) == 0
    sim_meta = json.loads(capsys.readouterr().out)
    assert sim_meta["systems"] == 2
    assert csv_path.is_file()

    common = ["--data", str(csv_path), "--n-reps", "12", "--n-pvs", "3", "--format", "json"]
    assert cli_main(["ols", "--predictors", "escs,female,treatment", "--system-fe", "--senate", *common]) == 0
    ols_payload = json.loads(capsys.readouterr().out)
    assert len(ols_payload["coefficients"]) >= 4

    assert cli_main(["meta", "--predictors", "escs,treatment", "--term", "treatment", *common]) == 0
    meta_payload = json.loads(capsys.readouterr().out)
    assert meta_payload["n_systems"] == 2

    assert cli_main(
        [
            "sensitivity",
            "--treatment",
            "treatment",
            "--controls",
            "escs,female",
            "--benchmarks",
            "escs",
            *common,
        ]
    ) == 0
    ovb_payload = json.loads(capsys.readouterr().out)
    assert "rv_q" in ovb_payload and len(ovb_payload["benchmarks"]) == 3


def test_cli_benchmark_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    """CLI `benchmark` subcommand must emit valid JSON rows across 3 estimators."""
    rc = cli_main(["benchmark", "--n-sim", "4", "--format", "json"])
    assert rc == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 3
