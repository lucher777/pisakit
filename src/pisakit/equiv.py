"""Cross-package and dense-matrix numerical equivalence verification."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats

from .datasets import load_pisa2022_micro_slice
from .models import _assemble_design_inputs, _solve_batched_wls, fit_pv_multilevel, fit_pv_ols
from .types import DesignSpec, EquivalenceReport
from .weights import normalize_weights


def _reference_scipy_wls_cube(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    w_all: np.ndarray,
) -> np.ndarray:
    """Solve all ``(1 + R) x M`` WLS regressions independently via ``scipy.linalg.lstsq``."""
    n_weights = w_all.shape[1]
    p_dim = x_mat.shape[1]
    m_pvs = y_mat.shape[1]
    beta_ref = np.empty((n_weights, p_dim, m_pvs), dtype=np.float64)
    for r_idx in range(n_weights):
        sw = np.sqrt(np.maximum(w_all[:, r_idx], 0.0))
        xw = x_mat * sw[:, None]
        for m_idx in range(m_pvs):
            yw = y_mat[:, m_idx] * sw
            coef, _, _, _ = linalg.lstsq(xw, yw)
            beta_ref[r_idx, :, m_idx] = coef
    return beta_ref


def _reference_oecd_fay_rubin(
    sub_df: pd.DataFrame,
    predictors: list[str],
    design: DesignSpec,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Compute exact OECD/intsvy Fay BRR + Barnard-Rubin statistics from scratch."""
    x_mat, _, y_mat, w_full, w_reps, df_com, _ = _assemble_design_inputs(
        data=sub_df,
        outcome_pattern="PV{k}MATH",
        predictors=predictors,
        design=design,
    )
    w_all = np.column_stack([w_full, w_reps])
    beta_ref = _reference_scipy_wls_cube(x_mat, y_mat, w_all)
    m_pvs = y_mat.shape[1]
    p_dim = x_mat.shape[1]
    n_reps = w_reps.shape[1]
    fay_mult = 1.0 / (n_reps * ((1.0 - design.fay_k) ** 2))

    q_m = beta_ref[0, :, :].T  # (M, p)
    u_m = np.empty((m_pvs, p_dim), dtype=np.float64)
    for m_idx in range(m_pvs):
        diff_r = beta_ref[1:, :, m_idx] - q_m[m_idx][None, :]
        u_m[m_idx] = fay_mult * np.sum(diff_r * diff_r, axis=0)

    q_bar = np.mean(q_m, axis=0)
    u_bar = np.mean(u_m, axis=0)
    b_mat = np.sum((q_m - q_bar[None, :]) ** 2, axis=0) / max(m_pvs - 1, 1)
    t_var = u_bar + (1.0 + 1.0 / m_pvs) * b_mat
    se_total = np.sqrt(t_var)
    se_sampling = np.sqrt(u_bar)
    se_imputation = np.sqrt((1.0 + 1.0 / m_pvs) * b_mat)

    df_br = np.empty(p_dim, dtype=np.float64)
    fmi = np.empty(p_dim, dtype=np.float64)
    for j in range(p_dim):
        r_j = ((1.0 + 1.0 / m_pvs) * b_mat[j]) / u_bar[j] if u_bar[j] > 0 else 0.0
        df_old = (m_pvs - 1.0) * ((1.0 + 1.0 / r_j) ** 2) if r_j > 0 else np.inf
        lam = (1.0 + 1.0 / m_pvs) * b_mat[j] / t_var[j]
        nu_obs = ((df_com + 1.0) / (df_com + 3.0)) * df_com * (1.0 - lam)
        df_star = (1.0 / (1.0 / df_old + 1.0 / nu_obs)) if np.isfinite(df_old) else nu_obs
        df_br[j] = df_star
        fmi[j] = (r_j + 2.0 / (df_star + 3.0)) / (r_j + 1.0)

    return {
        "estimate": q_bar,
        "se_total": se_total,
        "se_sampling": se_sampling,
        "se_imputation": se_imputation,
        "df": df_br,
        "fmi": fmi,
        "t_stat": q_bar / se_total,
        "p_value": 2.0 * stats.t.sf(np.abs(q_bar / se_total), df=df_br),
    }


def _reference_dense_reml_at_lambda(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    w_norm: np.ndarray,
    group_ids: np.ndarray,
    lam: float,
) -> tuple[float, np.ndarray, float, float, float]:
    """Evaluate exact two-level REML via explicit per-cluster dense covariance inversion."""
    unique_g = np.unique(group_ids)
    clusters = [np.where(group_ids == g)[0] for g in unique_g]
    p_dim = x_mat.shape[1]
    df_resid = max(float(np.sum(w_norm)) - p_dim, 1.0)

    xvx = np.zeros((p_dim, p_dim), dtype=np.float64)
    xvy = np.zeros(p_dim, dtype=np.float64)
    logdet_v_rel = 0.0
    v_invs: list[np.ndarray] = []

    for idx in clusters:
        wj = w_norm[idx]
        nj = len(idx)
        v_j = np.diag(1.0 / wj) + lam * np.ones((nj, nj), dtype=np.float64)
        v_inv = np.linalg.inv(v_j)
        v_invs.append(v_inv)
        _, ld = np.linalg.slogdet(v_j)
        logdet_v_rel += float(ld + np.sum(np.log(wj)))
        xj = x_mat[idx]
        yj = y_vec[idx]
        xvx += xj.T @ v_inv @ xj
        xvy += xj.T @ v_inv @ yj

    beta = np.linalg.solve(xvx, xvy)
    _, logdet_xvx = np.linalg.slogdet(xvx)
    rss = 0.0
    for idx, v_inv in zip(clusters, v_invs):
        ej = y_vec[idx] - x_mat[idx] @ beta
        rss += float(ej.T @ v_inv @ ej)
    sigma2 = max(rss, 1e-12) / df_resid
    tau2 = lam * sigma2
    icc = tau2 / max(tau2 + sigma2, 1e-12)
    obj = logdet_v_rel + float(logdet_xvx) + df_resid * float(np.log(max(sigma2, 1e-12)))
    return obj, beta, tau2, sigma2, icc


def _reference_dense_reml(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    w_norm: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Solve two-level weighted REML via independent dense covariance optimization."""
    opt = optimize.minimize_scalar(
        lambda ll: _reference_dense_reml_at_lambda(
            x_mat, y_vec, w_norm, group_ids, float(np.exp(ll))
        )[0],
        bounds=(-12.0, 6.0),
        method="bounded",
    )
    _, beta_hat, tau2_hat, sigma2_hat, icc_hat = _reference_dense_reml_at_lambda(
        x_mat, y_vec, w_norm, group_ids, float(np.exp(opt.x))
    )
    return beta_hat, tau2_hat, sigma2_hat, icc_hat


def verify_numerical_equivalence(
    data: pd.DataFrame | None = None,
    wls_tol: float = 1e-10,
    rubin_tol: float = 1e-11,
    reml_tol: float = 1e-9,
) -> EquivalenceReport:
    """Verify pisakit estimators against SciPy WLS, OECD Fay BRR, and dense REML.

    Evaluates three independent numerical cross-checks on the bundled real OECD
    PISA 2022 micro-slice (or a supplied PISA DataFrame):

    1. Batched rank-2 BLAS contraction vs element-wise ``scipy.linalg.lstsq``
       across all ``5 x 81 = 405`` weighted regressions.
    2. ``fit_pv_ols`` pooled estimates, standard errors, Barnard-Rubin degrees
       of freedom, and FMI vs an independent from-scratch Fay BRR + Rubin loop.
    3. ``fit_pv_multilevel`` cluster-Woodbury REML vs explicit dense block-diagonal
       covariance matrix inversion on single-system data.
    """
    df = load_pisa2022_micro_slice() if data is None else data
    spec5 = DesignSpec.from_preset("pisa", n_pvs=5, n_reps=80)

    x_mat, _, y_mat, w_full, w_reps, _, _ = _assemble_design_inputs(
        data=df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female", "hi_escs"],
        design=spec5,
        system_col="CNT",
        add_system_fe=True,
        senate_weights=True,
    )
    w_all = np.column_stack([w_full, w_reps])

    # Warmup then benchmark batched vs element-wise SciPy lstsq
    _ = _solve_batched_wls(x_mat, y_mat, w_all)
    t_b0 = time.perf_counter()
    beta_batched = _solve_batched_wls(x_mat, y_mat, w_all)
    batched_ms = (time.perf_counter() - t_b0) * 1000.0

    t_l0 = time.perf_counter()
    beta_ref = _reference_scipy_wls_cube(x_mat, y_mat, w_all)
    loop_ms = (time.perf_counter() - t_l0) * 1000.0
    wls_err = float(np.max(np.abs(beta_batched - beta_ref)))
    speedup = loop_ms / max(batched_ms, 1e-6)

    # Check Fay BRR + Rubin + Barnard-Rubin on FIN subset
    fin_df = df[df["CNT"] == "FIN"].copy()
    ols_fin = fit_pv_ols(fin_df, "PV{k}MATH", ["escs", "female"], spec5)
    ref_fin = _reference_oecd_fay_rubin(fin_df, ["escs", "female"], spec5)
    res_df = ols_fin.to_dataframe()
    brr_errs = [
        float(np.max(np.abs(res_df[col].to_numpy() - ref_fin[col])))
        for col in ("estimate", "se_total", "se_sampling", "se_imputation", "df", "fmi")
    ]
    brr_err = max(brr_errs)

    # Check Woodbury two-level REML vs explicit dense block covariance on FIN (PV1MATH, full weight)
    spec_1pv = DesignSpec(
        name="pisa_1pv",
        final_weight="W_FSTUWT",
        rep_prefix="W_FSTURWT",
        n_reps=2,
        variance_method="fay_brr",
        fay_k=0.5,
        n_pvs=1,
        stratum_col="STRATUM",
        psu_col="CNTSCHID",
    )
    fin_1pv = fin_df.copy()
    fin_1pv["W_FSTURWT1"] = fin_1pv["W_FSTUWT"]
    fin_1pv["W_FSTURWT2"] = fin_1pv["W_FSTUWT"]
    mlm_fin = fit_pv_multilevel(
        fin_1pv,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female"],
        cluster_col="CNTSCHID",
        design=spec_1pv,
    )
    w_norm = normalize_weights(fin_1pv, "W_FSTUWT", "CNTSCHID", method="mean_one")
    x_fin = np.column_stack(
        [np.ones(len(fin_1pv)), fin_1pv["escs"].to_numpy(), fin_1pv["female"].to_numpy()]
    )
    y_fin = fin_1pv["PV1MATH"].to_numpy(dtype=np.float64)
    gids_fin = fin_1pv["CNTSCHID"].to_numpy()
    lam_hat = mlm_fin.fit_stats["tau2_school"] / max(mlm_fin.fit_stats["sigma2_student"], 1e-12)
    _, beta_exact, tau2_exact, sig2_exact, icc_exact = _reference_dense_reml_at_lambda(
        x_fin, y_fin, w_norm, gids_fin, lam_hat
    )
    beta_opt, _, _, icc_opt = _reference_dense_reml(x_fin, y_fin, w_norm, gids_fin)
    reml_errs = [
        float(np.max(np.abs(mlm_fin.to_dataframe()["estimate"].to_numpy() - beta_exact))),
        abs(mlm_fin.fit_stats["tau2_school"] - tau2_exact),
        abs(mlm_fin.fit_stats["sigma2_student"] - sig2_exact),
        abs(mlm_fin.fit_stats["icc"] - icc_exact),
    ]
    reml_err = max(reml_errs)
    opt_agree = (
        float(np.max(np.abs(mlm_fin.to_dataframe()["estimate"].to_numpy() - beta_opt))) < 1e-6
        and abs(mlm_fin.fit_stats["icc"] - icc_opt) < 1e-6
    )

    passed = (wls_err < wls_tol) and (brr_err < rubin_tol) and (reml_err < reml_tol) and opt_agree
    return EquivalenceReport(
        wls_max_abs_err=wls_err,
        wls_batched_ms=batched_ms,
        wls_loop_ms=loop_ms,
        wls_speedup=speedup,
        brr_rubin_max_abs_err=brr_err,
        reml_woodbury_max_abs_err=reml_err,
        passed=passed,
    )
