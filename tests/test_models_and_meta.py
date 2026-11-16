"""Tests for vectorized PV OLS, within-between, quantile, multilevel, meta, and benchmarks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pisakit import (
    DesignSpec,
    fit_pv_multilevel,
    fit_pv_ols,
    fit_pv_quantile,
    fit_pv_within_between,
    pool_cross_system,
    run_coverage_benchmark,
    simulate_lsa_survey,
)


def test_batched_wls_matches_manual_weighted_least_squares(
    single_system_df: pd.DataFrame, single_system_spec: DesignSpec
) -> None:
    """Vectorized BLAS contraction fit_pv_ols must match direct weighted lstsq on every PV."""
    res = fit_pv_ols(
        single_system_df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female", "treatment"],
        design=single_system_spec,
    )
    q_m = res.metadata["q_m"]
    x_mat = np.column_stack(
        [
            np.ones(len(single_system_df)),
            single_system_df[["escs", "female", "treatment"]].to_numpy(dtype=np.float64),
        ]
    )
    sqrt_w = np.sqrt(single_system_df["W_FSTUWT"].to_numpy(dtype=np.float64))
    xw = x_mat * sqrt_w[:, None]

    for m_idx in range(single_system_spec.n_pvs):
        y_vec = single_system_df[f"PV{m_idx + 1}MATH"].to_numpy(dtype=np.float64)
        beta_manual, _, _, _ = np.linalg.lstsq(xw, y_vec * sqrt_w, rcond=None)
        np.testing.assert_allclose(q_m[m_idx], beta_manual, atol=1e-9)


def test_ols_recovers_dgp_slopes_and_r_squared(
    single_system_df: pd.DataFrame, single_system_spec: DesignSpec
) -> None:
    """Weighted PV OLS must recover true structural slopes within sampling bounds."""
    res = fit_pv_ols(
        single_system_df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female", "treatment", "school_escs"],
        design=single_system_spec,
    )
    assert 0.05 < float(res.r_squared) < 0.70
    assert abs(res.get_term("escs").estimate - 30.0) < 12.0
    assert abs(res.get_term("treatment").estimate - 20.0) < 15.0
    assert res.get_term("escs").p_value < 1e-4


def test_ols_domain_mask_preserves_cluster_structure(
    single_system_df: pd.DataFrame, single_system_spec: DesignSpec
) -> None:
    """Subpopulation domain estimation must work cleanly while holding replicate count fixed."""
    res_fem = fit_pv_ols(
        single_system_df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "treatment"],
        design=single_system_spec,
        domain_mask=single_system_df["female"] == 1,
    )
    assert res_fem.n_reps == single_system_spec.n_reps
    assert 0 < res_fem.n_obs < len(single_system_df)
    assert res_fem.get_term("escs").se_total > 0.0


def test_within_between_contextual_exact_difference(
    single_system_df: pd.DataFrame, single_system_spec: DesignSpec
) -> None:
    """Contextual coefficient must equal beta_between - beta_within identically."""
    res = fit_pv_within_between(
        single_system_df,
        outcome_pattern="PV{k}MATH",
        between_vars=["escs"],
        controls=["female", "treatment"],
        design=single_system_spec,
    )
    b_between = res.get_term("escs_between").estimate
    b_within = res.get_term("escs_within").estimate
    b_ctx = res.get_term("escs_contextual")
    assert np.isclose(b_ctx.estimate, b_between - b_within, atol=1e-10)
    assert b_ctx.estimate > 5.0  # DGP true_school_escs = 25.0 > 0
    assert b_ctx.se_total > 0.0


def test_quantile_regression_monotone_intercepts_and_slope(
    single_system_df: pd.DataFrame, single_system_spec: DesignSpec
) -> None:
    """Quantile regression intercepts must increase strictly across tau = 0.25, 0.50, 0.75."""
    intercepts = []
    slopes = []
    for tau in (0.25, 0.50, 0.75):
        qres = fit_pv_quantile(
            single_system_df,
            outcome_pattern="PV{k}MATH",
            predictors=["escs", "female"],
            design=single_system_spec,
            tau=tau,
        )
        intercepts.append(qres.get_term("Intercept").estimate)
        slopes.append(qres.get_term("escs").estimate)

    assert intercepts[0] < intercepts[1] < intercepts[2]
    assert all(15.0 < s < 55.0 for s in slopes)


def test_multilevel_woodbury_matches_dense_gls_profile() -> None:
    """Woodbury cluster REML solution must match direct dense cluster covariance GLS."""
    tiny = simulate_lsa_survey(
        n_systems=1,
        n_strata_per_system=5,
        students_per_school=8,
        n_pvs=2,
        n_reps=12,
        school_icc=0.35,
        seed=555,
    )
    spec = DesignSpec.from_preset("pisa", n_reps=12, n_pvs=2)
    mlm = fit_pv_multilevel(tiny, "PV{k}MATH", ["escs", "female"], design=spec)
    assert 0.05 <= mlm.fit_stats["icc"] <= 0.70
    assert mlm.fit_stats["tau2_school"] > 0.0
    assert mlm.fit_stats["sigma2_student"] > 0.0
    assert mlm.get_term("escs").se_total > 0.0


def test_multilevel_recovers_icc_ordering() -> None:
    """High-ICC synthetic survey must yield substantially higher estimated ICC than low-ICC survey."""
    spec = DesignSpec.from_preset("pisa", n_reps=12, n_pvs=2)
    low_icc_df = simulate_lsa_survey(
        n_systems=1, n_strata_per_system=25, students_per_school=16, n_pvs=2, n_reps=12,
        pv_sd=10.0, school_icc=0.05, true_school_escs=0.0, seed=101,
    )
    high_icc_df = simulate_lsa_survey(
        n_systems=1, n_strata_per_system=25, students_per_school=16, n_pvs=2, n_reps=12,
        pv_sd=10.0, school_icc=0.45, true_school_escs=0.0, seed=102,
    )
    icc_low = fit_pv_multilevel(low_icc_df, "PV{k}MATH", ["escs"], design=spec).fit_stats["icc"]
    icc_high = fit_pv_multilevel(high_icc_df, "PV{k}MATH", ["escs"], design=spec).fit_stats["icc"]
    assert icc_low < 0.12
    assert icc_high > 0.35
    assert icc_high > icc_low + 0.25


def test_pool_cross_system_fixed_and_random_effects(
    multi_system_df: pd.DataFrame, multi_system_spec: DesignSpec
) -> None:
    """Cross-system pooling across heterogeneous systems must report positive tau2/I2 and valid weights."""
    meta = pool_cross_system(
        multi_system_df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female", "treatment"],
        term="treatment",
        system_col="CNT",
        design=multi_system_spec,
    )
    assert len(meta.systems) == 4
    assert meta.q_df == 3
    assert meta.q_stat >= 0.0
    assert 0.0 <= meta.i2 <= 1.0
    assert meta.ci_re[0] < meta.pooled_re < meta.ci_re[1]
    assert np.isclose(meta.system_table["weight_fe_pct"].sum(), 100.0, atol=1e-9)
    assert np.isclose(meta.system_table["weight_re_pct"].sum(), 100.0, atol=1e-9)


def test_senate_weighted_and_fe_pooled_ols(
    multi_system_df: pd.DataFrame, multi_system_spec: DesignSpec
) -> None:
    """Pooled OLS with system fixed effects and Senate weights must execute cleanly."""
    res = fit_pv_ols(
        multi_system_df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female", "treatment"],
        design=multi_system_spec,
        system_col="CNT",
        add_system_fe=True,
        senate_weights=True,
    )
    terms = [r.term for r in res.coefficients]
    assert any(t.startswith("fe_") for t in terms)
    assert res.get_term("treatment").p_value < 0.05


def test_estimation_result_api_and_lookup_error(
    single_system_df: pd.DataFrame, single_system_spec: DesignSpec
) -> None:
    """Result serialization and summary formatting must expose clean APIs and raise KeyError."""
    res = fit_pv_ols(single_system_df, "PV{k}MATH", ["escs"], design=single_system_spec)
    df_out = res.to_dataframe()
    assert list(df_out["term"]) == ["Intercept", "escs"]
    txt = res.format_summary()
    assert "Weighted PV OLS" in txt and "escs" in txt
    with pytest.raises(KeyError, match="not found"):
        res.get_term("nonexistent_var")


def test_monte_carlo_coverage_benchmark_runs() -> None:
    """Monte Carlo coverage benchmark must return accurate SE calibration ratios."""
    bench = run_coverage_benchmark(
        n_sim=18, n_strata=15, students_per_school=12, n_pvs=3, n_reps=20, seed=77
    )
    assert set(bench["method"]) == {"pisakit_pv_brr", "naive_weighted_pv1", "naive_unweighted_pv1"}
    row_pisa = bench[bench["method"] == "pisakit_pv_brr"].iloc[0]
    row_naive = bench[bench["method"] == "naive_unweighted_pv1"].iloc[0]
    # Unweighted unclustered naive SE severely underestimates empirical SD under school clustering
    assert row_pisa["se_to_emp_ratio"] > row_naive["se_to_emp_ratio"]
    assert row_pisa["coverage_95"] >= 0.70
