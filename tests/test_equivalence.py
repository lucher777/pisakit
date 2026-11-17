"""Cross-package numerical equivalence and bundled PISA 2022 data tests."""
from __future__ import annotations

import numpy as np

from pisakit import (
    DesignSpec,
    cli,
    fit_pv_ols,
    load_pisa2022_micro_slice,
    load_pisa2022_system_summary,
    verify_numerical_equivalence,
)
from pisakit.equiv import _reference_oecd_fay_rubin, _reference_scipy_wls_cube
from pisakit.models import _assemble_design_inputs, _solve_batched_wls


def test_pisa2022_micro_slice_integrity() -> None:
    """Bundled OECD PISA 2022 micro-slice has expected shape, weights, and systems."""
    df = load_pisa2022_micro_slice()
    assert len(df) == 960
    assert set(df["CNT"].unique()) == {"FIN", "JPN", "KOR", "EST"}
    assert df["CNTSCHID"].nunique() == 60
    assert all(f"PV{k}MATH" in df.columns for k in range(1, 6))
    assert all(f"W_FSTURWT{r}" in df.columns for r in range(1, 81))
    assert np.all(df["W_FSTUWT"] > 0.0)
    assert np.all(np.isfinite(df["escs"]))


def test_pisa2022_15cnt_summary_integrity() -> None:
    """Bundled 15-system full-sample summary contains valid gradients and bounds."""
    summary = load_pisa2022_system_summary()
    assert len(summary) == 15
    assert {"JPN", "KOR", "EST", "FIN", "DEU", "USA", "MEX"}.issubset(set(summary["cnt"]))
    assert np.all(summary["n_students"] > 4000)
    assert np.all(summary["beta_escs"] > 15.0)
    assert np.all(summary["beta_between"] > summary["beta_within"])
    assert np.all((summary["rv_q"] > 0.5) & (summary["rv_q"] < 1.0))


def test_batched_wls_matches_scipy_lstsq_across_405_regressions() -> None:
    """Rank-2 BLAS contraction matches scipy.linalg.lstsq within < 1e-10 across 405 fits."""
    df = load_pisa2022_micro_slice()
    spec = DesignSpec.from_preset("pisa", n_pvs=5, n_reps=80)
    x_mat, _, y_mat, w_full, w_reps, _, _ = _assemble_design_inputs(
        data=df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female", "hi_escs"],
        design=spec,
        system_col="CNT",
        add_system_fe=True,
        senate_weights=True,
    )
    w_all = np.column_stack([w_full, w_reps])
    beta_batched = _solve_batched_wls(x_mat, y_mat, w_all)
    beta_scipy = _reference_scipy_wls_cube(x_mat, y_mat, w_all)
    assert beta_batched.shape == (81, 7, 5)
    assert np.max(np.abs(beta_batched - beta_scipy)) < 1e-10


def test_fay_brr_rubin_matches_oecd_reference_on_fin_slice() -> None:
    """fit_pv_ols matches from-scratch OECD Fay BRR + Barnard-Rubin pooling < 1e-11."""
    df = load_pisa2022_micro_slice()
    fin = df[df["CNT"] == "FIN"].copy()
    spec = DesignSpec.from_preset("pisa", n_pvs=5, n_reps=80)
    res = fit_pv_ols(fin, "PV{k}MATH", ["escs", "female"], spec).to_dataframe()
    ref = _reference_oecd_fay_rubin(fin, ["escs", "female"], spec)
    for col in ("estimate", "se_total", "se_sampling", "se_imputation", "df", "fmi", "t_stat", "p_value"):
        assert np.max(np.abs(res[col].to_numpy() - ref[col])) < 1e-11


def test_verify_numerical_equivalence_suite_passes() -> None:
    """Full equivalence suite passes WLS, Fay BRR, and dense Woodbury REML checks."""
    report = verify_numerical_equivalence()
    assert report.passed is True
    assert report.wls_max_abs_err < 1e-10
    assert report.brr_rubin_max_abs_err < 1e-11
    assert report.reml_woodbury_max_abs_err < 1e-9
    assert np.isfinite(report.wls_speedup) and report.wls_speedup > 0.0
    assert "PASS" in report.summary()


def test_cli_equiv_and_demo_real_exit_zero(capsys) -> None:
    """CLI equiv and demo-real subcommands execute end-to-end and return exit 0."""
    rc_eq = cli.main(["equiv", "--format", "json"])
    assert rc_eq == 0
    out_eq = capsys.readouterr().out
    assert '"passed": true' in out_eq

    rc_real = cli.main(["demo-real", "--format", "json"])
    assert rc_real == 0
    out_real = capsys.readouterr().out
    assert '"n_slice": 960' in out_real
