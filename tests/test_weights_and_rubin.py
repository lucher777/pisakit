"""Tests for survey weights, Hadamard BRR design, Rubin rules, and descriptives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from pisakit import (
    DesignSpec,
    build_hadamard_matrix,
    combine_rubin,
    compute_senate_weights,
    descriptive_pv_stats,
    effective_sample_size,
    generate_fay_brr_matrix,
    get_design_preset,
    joint_wald_test,
    normalize_weights,
    prepare_domain_weights,
    replicate_variance,
    resolve_pv_columns,
    resolve_rep_columns,
    validate_survey_frame,
)


@pytest.mark.parametrize("order", [2, 4, 8, 12, 20, 40, 80])
def test_hadamard_orthogonality_across_orders(order: int) -> None:
    """Paley and Sylvester Hadamard matrices must satisfy H^T H = n * I."""
    h_mat = build_hadamard_matrix(order)
    assert h_mat.shape == (order, order)
    assert set(np.unique(h_mat)).issubset({-1.0, 1.0})
    gram = h_mat.T @ h_mat
    np.testing.assert_allclose(gram, order * np.eye(order), atol=1e-12)


@pytest.mark.parametrize("fay_k", [0.0, 0.3, 0.5])
def test_fay_brr_matrix_multipliers_and_zero_mean(fay_k: float) -> None:
    """Replicate multipliers must average to 1 across replicates and match 1 +/- (1-k)."""
    mult = generate_fay_brr_matrix(n_strata=15, n_reps=40, fay_k=fay_k)
    assert mult.shape == (15, 2, 40)
    np.testing.assert_allclose(np.mean(mult, axis=2), 1.0, atol=1e-12)
    expected_vals = {round(1.0 + (1.0 - fay_k), 8), round(1.0 - (1.0 - fay_k), 8)}
    observed_vals = {round(float(v), 8) for v in np.unique(mult)}
    assert observed_vals == expected_vals


@pytest.mark.parametrize("fay_k", [0.0, 0.3, 0.5])
def test_replicate_variance_matches_exact_paired_stratum_total(fay_k: float) -> None:
    """Fay BRR replicate variance on a linear total must equal exact sum (w0 y0 - w1 y1)^2."""
    n_strata = 6
    n_reps = 20
    mult = generate_fay_brr_matrix(n_strata=n_strata, n_reps=n_reps, fay_k=fay_k)
    rng = np.random.default_rng(77)
    psu_totals = rng.uniform(50.0, 200.0, size=(n_strata, 2))
    theta_full = float(np.sum(psu_totals))

    theta_reps = np.array(
        [np.sum(psu_totals * mult[:, :, r]) for r in range(n_reps)], dtype=np.float64
    )
    brr_var = float(replicate_variance(theta_full, theta_reps, method="fay_brr", fay_k=fay_k))
    exact_var = float(np.sum((psu_totals[:, 0] - psu_totals[:, 1]) ** 2))
    assert np.isclose(brr_var, exact_var, rtol=1e-10)


def test_design_presets_and_validation_guards() -> None:
    """Built-in presets and invalid DesignSpec arguments must validate strictly."""
    pisa = get_design_preset("pisa")
    timss = get_design_preset("timss")
    pirls = get_design_preset("pirls")
    assert pisa.variance_method == "fay_brr" and pisa.fay_k == 0.5
    assert timss.variance_method == "jk2" and timss.n_reps == 75
    assert pirls.final_weight == "TOTWGT"

    with pytest.raises(ValueError, match="fay_k"):
        DesignSpec(fay_k=1.0)
    with pytest.raises(ValueError, match="n_reps"):
        DesignSpec(n_reps=1)
    with pytest.raises(ValueError, match="Unknown design preset"):
        get_design_preset("naep_unknown")


def test_resolve_pv_and_rep_columns_variants() -> None:
    """PV column resolver must support both {k} placeholders and prefix+digits."""
    df = pd.DataFrame(
        {
            "W_FSTUWT": [10.0, 12.0],
            "W_FSTURWT1": [15.0, 6.0],
            "W_FSTURWT2": [5.0, 18.0],
            "PV1MATH": [500.0, 510.0],
            "PV2MATH": [505.0, 512.0],
            "ASMMA01": [480.0, 490.0],
            "ASMMA02": [485.0, 495.0],
        }
    )
    assert resolve_pv_columns(df, "PV{k}MATH", n_pvs=2) == ["PV1MATH", "PV2MATH"]
    assert resolve_pv_columns(df, "ASMMA", n_pvs=2) == ["ASMMA01", "ASMMA02"]

    spec = DesignSpec.from_preset("pisa", n_reps=2, n_pvs=2)
    assert resolve_rep_columns(df, spec) == ["W_FSTURWT1", "W_FSTURWT2"]
    with pytest.raises(KeyError):
        resolve_pv_columns(df, "PV{k}READ", n_pvs=2)


def test_validate_survey_frame_rejects_corrupted_weights(single_system_df: pd.DataFrame, single_system_spec: DesignSpec) -> None:
    """Validation must reject negative final weights, NaNs, and missing columns."""
    bad = single_system_df.copy()
    bad.loc[0, "W_FSTUWT"] = -1.0
    with pytest.raises(ValueError, match="negative weights"):
        validate_survey_frame(bad, single_system_spec, outcome_pattern="PV{k}MATH")

    bad2 = single_system_df.copy()
    bad2.loc[0, "PV1MATH"] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_survey_frame(bad2, single_system_spec, outcome_pattern="PV{k}MATH")


def test_senate_weights_equalize_systems_and_preserve_ratios(
    multi_system_df: pd.DataFrame, multi_system_spec: DesignSpec
) -> None:
    """Senate scaling must equalize weighted totals while leaving W_rep/W_full unchanged."""
    scaled = compute_senate_weights(
        multi_system_df, multi_system_spec, system_col="CNT", target_sum=500.0
    )
    sys_sums = scaled.groupby("CNT")["W_FSTUWT"].sum()
    np.testing.assert_allclose(sys_sums.to_numpy(), 500.0, atol=1e-9)

    orig_ratio = multi_system_df["W_FSTURWT1"] / multi_system_df["W_FSTUWT"]
    new_ratio = scaled["W_FSTURWT1"] / scaled["W_FSTUWT"]
    np.testing.assert_allclose(new_ratio.to_numpy(), orig_ratio.to_numpy(), rtol=1e-12)


def test_normalize_weights_and_effective_sample_size(single_system_df: pd.DataFrame) -> None:
    """Within-school normalized weights must hit exact cluster targets and Kish N_eff <= N."""
    w_mean1 = normalize_weights(single_system_df, "W_FSTUWT", group_col="CNTSCHID", method="mean_one")
    chk = single_system_df.assign(wn=w_mean1).groupby("CNTSCHID")["wn"].mean()
    np.testing.assert_allclose(chk.to_numpy(), 1.0, atol=1e-11)

    w_sum1 = normalize_weights(single_system_df, "W_FSTUWT", group_col="CNTSCHID", method="sum_one")
    chk_s = single_system_df.assign(ws=w_sum1).groupby("CNTSCHID")["ws"].sum()
    np.testing.assert_allclose(chk_s.to_numpy(), 1.0, atol=1e-11)

    n_eff, deff = effective_sample_size(single_system_df["W_FSTUWT"].to_numpy())
    assert 0.0 < n_eff < len(single_system_df)
    assert deff > 1.0


def test_domain_weights_zero_excluded_rows(
    single_system_df: pd.DataFrame, single_system_spec: DesignSpec
) -> None:
    """Domain weight masking must zero out non-domain rows without dropping observations."""
    mask = single_system_df["female"] == 1
    w_f, w_r = prepare_domain_weights(single_system_df, mask, single_system_spec)
    assert w_f.shape == (len(single_system_df),)
    assert w_r.shape == (len(single_system_df), single_system_spec.n_reps)
    assert np.all(w_f[~mask.to_numpy()] == 0.0)
    assert np.all(w_r[~mask.to_numpy(), :] == 0.0)
    assert np.all(w_f[mask.to_numpy()] > 0.0)


def test_combine_rubin_and_barnard_rubin_analytical_check() -> None:
    """Rubin + Barnard-Rubin pooling must match explicit analytical hand calculation."""
    q_m = np.array([[10.0], [12.0], [11.0], [13.0], [9.0]])
    u_m = np.array([[4.0], [4.2], [3.8], [4.1], [3.9]])
    df_com = 40.0

    q_bar_exp = float(np.mean(q_m))
    u_bar_exp = float(np.mean(u_m))
    b_exp = float(np.var(q_m[:, 0], ddof=1))
    m = 5.0
    t_exp = u_bar_exp + (1.0 + 1.0 / m) * b_exp
    r_exp = (1.0 + 1.0 / m) * b_exp / u_bar_exp
    nu_old = (m - 1.0) * (1.0 + 1.0 / r_exp) ** 2
    lam = (1.0 + 1.0 / m) * b_exp / t_exp
    nu_obs = ((df_com + 1.0) / (df_com + 3.0)) * df_com * (1.0 - lam)
    nu_star = 1.0 / (1.0 / nu_old + 1.0 / nu_obs)

    res, decomps = combine_rubin(q_m, u_m, df_complete=df_com, alpha=0.05)
    assert np.isclose(res["q_bar"][0], q_bar_exp)
    assert np.isclose(res["u_bar"][0], u_bar_exp)
    assert np.isclose(res["b_imputation"][0], b_exp)
    assert np.isclose(res["total_var"][0], t_exp)
    assert np.isclose(res["df_rubin"][0], nu_old)
    assert np.isclose(res["df_br"][0], nu_star)
    assert len(decomps) == 1
    t_crit = float(stats.t.ppf(0.975, df=nu_star))
    assert np.isclose(res["ci_high"][0] - res["ci_low"][0], 2.0 * t_crit * np.sqrt(t_exp))


def test_joint_wald_d1_test_null_vs_signal() -> None:
    """Joint D1 Wald test must reject strong signal blocks and retain near-zero null blocks."""
    rng = np.random.default_rng(11)
    q_null = rng.normal(0.0, 0.05, size=(5, 2))
    cov_base = np.broadcast_to(np.diag([0.5, 0.5]), (5, 2, 2)).copy()
    null_res = joint_wald_test(q_null, cov_base)
    assert null_res["p_value"] > 0.50

    q_sig = q_null + np.array([3.5, -2.8])[None, :]
    sig_res = joint_wald_test(q_sig, cov_base)
    assert sig_res["d1_stat"] > 10.0
    assert sig_res["p_value"] < 0.01


def test_descriptive_pv_stats_and_gender_gap(
    single_system_df: pd.DataFrame, single_system_spec: DesignSpec
) -> None:
    """Weighted descriptive stats and between-group gaps must align with true gap direction."""
    overall = descriptive_pv_stats(
        single_system_df, "PV{k}MATH", single_system_spec, quantiles=(0.25, 0.5, 0.75)
    )
    mean_row = overall.get_term("mean")
    assert 470.0 < mean_row.estimate < 530.0
    assert overall.get_term("q25").estimate < overall.get_term("q50").estimate < overall.get_term("q75").estimate

    by_sex = descriptive_pv_stats(
        single_system_df,
        "PV{k}MATH",
        single_system_spec,
        by="female",
        quantiles=(0.5,),
        reference_group=0,
    )
    gap_mean = by_sex.get_term("diff(1-0):mean")
    w_all = single_system_df["W_FSTUWT"].to_numpy()
    fem_mask = (single_system_df["female"] == 1).to_numpy()
    pv_cols = [f"PV{k}MATH" for k in range(1, single_system_spec.n_pvs + 1)]
    manual_diff = float(
        np.mean(
            [
                np.average(single_system_df[c].to_numpy()[fem_mask], weights=w_all[fem_mask])
                - np.average(single_system_df[c].to_numpy()[~fem_mask], weights=w_all[~fem_mask])
                for c in pv_cols
            ]
        )
    )
    assert np.isclose(gap_mean.estimate, manual_diff, atol=1e-10)
    assert gap_mean.se_total > 0.0
    assert gap_mean.ci_low <= -8.0 <= gap_mean.ci_high
