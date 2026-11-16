"""
Rubin (1987), Barnard-Rubin (1999), and Li-Raghunathan-Rubin (1991)
combining rules and survey descriptive estimators across plausible values.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .types import (
    CoefficientRow,
    DesignSpec,
    EstimationResult,
    VarianceDecomposition,
)
from .weights import prepare_domain_weights, replicate_variance, validate_survey_frame


def combine_rubin(
    q_m: np.ndarray,
    u_m: np.ndarray,
    df_complete: float | None = None,
    alpha: float = 0.05,
) -> tuple[dict[str, np.ndarray], list[VarianceDecomposition]]:
    """Combine scalar or vector parameter estimates across ``M`` plausible values.

    Implements Rubin's (1987) combining rules with Barnard & Rubin's (1999)
    small-sample degrees-of-freedom adjustment when ``df_complete`` is supplied.

    Args:
        q_m: Point estimates across plausible values, shape ``(M, p)`` or ``(M,)``.
        u_m: Within-imputation sampling variances, shape ``(M, p)`` or ``(M,)``.
        df_complete: Complete-data degrees of freedom ``nu_com``. If ``None``,
            unadjusted Rubin degrees of freedom are used.
        alpha: Significance level for two-sided confidence intervals.

    Returns:
        A tuple ``(summary_arrays, decompositions)`` containing vectorized arrays
        and per-term ``VarianceDecomposition`` objects.
    """
    q_arr = np.asarray(q_m, dtype=np.float64)
    u_arr = np.asarray(u_m, dtype=np.float64)
    if q_arr.ndim == 1:
        q_arr = q_arr[:, None]
    if u_arr.ndim == 1:
        u_arr = u_arr[:, None]
    if q_arr.shape != u_arr.shape:
        raise ValueError(f"q_m shape {q_arr.shape} does not match u_m shape {u_arr.shape}.")

    m_pvs, p_dim = q_arr.shape
    if m_pvs < 1:
        raise ValueError("At least one plausible value is required.")

    q_bar = np.mean(q_arr, axis=0)
    u_bar = np.mean(u_arr, axis=0)

    if m_pvs > 1:
        diff = q_arr - q_bar[None, :]
        b_mat = np.sum(diff * diff, axis=0) / (m_pvs - 1.0)
    else:
        b_mat = np.zeros(p_dim, dtype=np.float64)

    imputation_inflate = 1.0 + (1.0 / m_pvs)
    total_var = u_bar + imputation_inflate * b_mat
    se_total = np.sqrt(np.maximum(total_var, 0.0))
    se_sampling = np.sqrt(np.maximum(u_bar, 0.0))
    se_imputation = np.sqrt(np.maximum(imputation_inflate * b_mat, 0.0))

    rvi = np.zeros(p_dim, dtype=np.float64)
    df_rubin = np.full(p_dim, np.inf, dtype=np.float64)
    df_br = np.full(p_dim, np.inf, dtype=np.float64)
    fmi = np.zeros(p_dim, dtype=np.float64)
    df_com_val = float(df_complete) if df_complete is not None else np.inf

    for j in range(p_dim):
        u_j = float(u_bar[j])
        b_j = float(b_mat[j])
        t_j = float(total_var[j])

        if u_j > 0.0 and b_j > 0.0:
            r_j = (imputation_inflate * b_j) / u_j
            rvi[j] = r_j
            if m_pvs > 1:
                df_old_j = (m_pvs - 1.0) * ((1.0 + (1.0 / r_j)) ** 2)
            else:
                df_old_j = np.inf
        elif b_j == 0.0:
            r_j = 0.0
            df_old_j = np.inf
        else:
            r_j = np.inf
            rvi[j] = r_j
            df_old_j = float(max(m_pvs - 1, 1))

        df_rubin[j] = df_old_j
        lam_j = (imputation_inflate * b_j / t_j) if t_j > 0.0 else 0.0

        if df_complete is not None and np.isfinite(df_com_val) and df_com_val > 0.0:
            nu_obs_j = (
                ((df_com_val + 1.0) / (df_com_val + 3.0))
                * df_com_val
                * max(1.0 - lam_j, 1e-12)
            )
            if np.isfinite(df_old_j):
                df_star_j = 1.0 / ((1.0 / df_old_j) + (1.0 / nu_obs_j))
            else:
                df_star_j = nu_obs_j
        else:
            df_star_j = df_old_j

        df_br[j] = df_star_j
        if np.isfinite(r_j):
            if np.isfinite(df_star_j):
                fmi[j] = (r_j + 2.0 / (df_star_j + 3.0)) / (r_j + 1.0)
            else:
                fmi[j] = r_j / (r_j + 1.0)
        else:
            fmi[j] = 1.0

    t_stat = np.divide(q_bar, se_total, out=np.zeros_like(q_bar), where=se_total > 0.0)
    p_value = np.ones(p_dim, dtype=np.float64)
    ci_low = np.empty(p_dim, dtype=np.float64)
    ci_high = np.empty(p_dim, dtype=np.float64)

    for j in range(p_dim):
        df_j = float(df_br[j])
        if se_total[j] <= 0.0:
            p_value[j] = 1.0 if q_bar[j] == 0.0 else 0.0
            ci_low[j] = q_bar[j]
            ci_high[j] = q_bar[j]
            continue

        if np.isfinite(df_j):
            p_value[j] = float(2.0 * stats.t.sf(np.abs(t_stat[j]), df=df_j))
            t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_j))
        else:
            p_value[j] = float(2.0 * stats.norm.sf(np.abs(t_stat[j])))
            t_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
        ci_low[j] = q_bar[j] - t_crit * se_total[j]
        ci_high[j] = q_bar[j] + t_crit * se_total[j]

    summary_arrays = {
        "q_bar": q_bar,
        "u_bar": u_bar,
        "b_imputation": b_mat,
        "total_var": total_var,
        "se_total": se_total,
        "se_sampling": se_sampling,
        "se_imputation": se_imputation,
        "rvi": rvi,
        "fmi": fmi,
        "df_rubin": df_rubin,
        "df_br": df_br,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

    decomps = [
        VarianceDecomposition(
            q_bar=float(q_bar[j]),
            u_bar=float(u_bar[j]),
            b_imputation=float(b_mat[j]),
            total_var=float(total_var[j]),
            se_total=float(se_total[j]),
            se_sampling=float(se_sampling[j]),
            se_imputation=float(se_imputation[j]),
            rvi=float(rvi[j]),
            fmi=float(fmi[j]),
            df_rubin=float(df_rubin[j]),
            df_complete=df_com_val,
            df_br=float(df_br[j]),
        )
        for j in range(p_dim)
    ]
    return summary_arrays, decomps


def joint_wald_test(
    q_m: np.ndarray,
    cov_m: np.ndarray,
    term_indices: Sequence[int] | None = None,
) -> dict[str, float]:
    """Li-Raghunathan-Rubin (1991) D1 multi-parameter Wald test across PVs.

    Tests the joint null hypothesis ``H0: theta_S = 0`` for a subset of
    parameters indexed by ``term_indices``.
    """
    q_arr = np.asarray(q_m, dtype=np.float64)
    cov_arr = np.asarray(cov_m, dtype=np.float64)
    m_pvs, p_dim = q_arr.shape
    if cov_arr.shape != (m_pvs, p_dim, p_dim):
        raise ValueError(
            f"cov_m must have shape ({m_pvs}, {p_dim}, {p_dim}), got {cov_arr.shape}."
        )

    idx = list(range(p_dim)) if term_indices is None else list(term_indices)
    k = len(idx)
    if k < 1:
        raise ValueError("term_indices must contain at least one index.")

    q_sub = q_arr[:, idx]
    cov_sub = cov_arr[:, np.ix_(idx, idx)[0], np.ix_(idx, idx)[1]]

    q_bar = np.mean(q_sub, axis=0)
    u_bar = np.mean(cov_sub, axis=0)
    u_inv = np.linalg.pinv(u_bar)

    if m_pvs > 1:
        diff = q_sub - q_bar[None, :]
        b_mat = (diff.T @ diff) / (m_pvs - 1.0)
        r_m = float((1.0 + 1.0 / m_pvs) * np.trace(b_mat @ u_inv) / k)
    else:
        r_m = 0.0

    d1_stat = float((q_bar.T @ u_inv @ q_bar) / (k * (1.0 + r_m)))
    t_df = k * (m_pvs - 1)
    if r_m <= 0.0 or m_pvs == 1:
        df_denom = 1e6
    elif t_df > 4:
        df_denom = 4.0 + (t_df - 4.0) * ((1.0 + (1.0 - 2.0 / t_df) / r_m) ** 2)
    else:
        df_denom = 0.5 * t_df * ((1.0 + 1.0 / k) * ((1.0 + 1.0 / r_m) ** 2))

    p_val = float(stats.f.sf(d1_stat, dfn=k, dfd=df_denom))
    return {
        "d1_stat": d1_stat,
        "df1": float(k),
        "df2": float(df_denom),
        "r_m": r_m,
        "p_value": p_val,
    }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Compute weighted sample quantiles via linear interpolation of EDF."""
    pos = weights > 0.0
    v = values[pos]
    w = weights[pos]
    order = np.argsort(v, kind="mergesort")
    v_sorted = v[order]
    w_sorted = w[order]
    cum_w = np.cumsum(w_sorted) - 0.5 * w_sorted
    cum_w /= float(np.sum(w_sorted))
    return np.interp(probs, cum_w, v_sorted)


def _compute_group_stat_matrix(
    y_mat: np.ndarray,
    w_all: np.ndarray,
    quantiles: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute weighted mean, SD, and quantiles for all ``(1 + R) x M`` pairs."""
    n_weights = w_all.shape[1]
    n_pvs = y_mat.shape[1]
    n_q = len(quantiles)
    n_stats = 2 + n_q
    probs = np.asarray(quantiles, dtype=np.float64)

    out = np.empty((n_weights, n_pvs, n_stats), dtype=np.float64)
    w_sums = np.sum(w_all, axis=0)  # (n_weights,)
    means = (w_all.T @ y_mat) / w_sums[:, None]  # (n_weights, M)
    out[:, :, 0] = means

    # Weighted variance and quantiles
    for pv_idx in range(n_pvs):
        y_vec = y_mat[:, pv_idx]
        resid_sq = (y_vec[None, :] - means[:, pv_idx][:, None]) ** 2
        var_vec = np.sum(w_all.T * resid_sq, axis=1) / w_sums
        out[:, pv_idx, 1] = np.sqrt(np.maximum(var_vec, 0.0))
        if n_q > 0:
            for r_idx in range(n_weights):
                out[r_idx, pv_idx, 2:] = _weighted_quantile(y_vec, w_all[:, r_idx], probs)
    return out, np.asarray(
        ["mean", "sd"] + [f"q{int(round(q * 100)):02d}" for q in quantiles],
        dtype=object,
    )


def descriptive_pv_stats(
    data: pd.DataFrame,
    outcome_pattern: str,
    design: DesignSpec,
    by: str | None = None,
    quantiles: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    reference_group: object | None = None,
    alpha: float = 0.05,
) -> EstimationResult:
    """Estimate survey-weighted means, SDs, quantiles, and optional group gaps.

    Computes all statistics across all ``M`` plausible values and ``R`` replicate
    weights, returning Rubin + Barnard-Rubin pooled estimates and SEs.
    """
    q_tuple = tuple(float(q) for q in quantiles)
    for q in q_tuple:
        if not (0.0 < q < 1.0):
            raise ValueError(f"Quantiles must lie strictly in (0, 1), got {q}")

    req = [by] if by is not None else []
    pv_cols, rep_cols = validate_survey_frame(
        data, design, outcome_pattern=outcome_pattern, required_cols=req
    )
    y_mat = data[pv_cols].to_numpy(dtype=np.float64)
    w_full = data[design.final_weight].to_numpy(dtype=np.float64)
    w_reps = data[rep_cols].to_numpy(dtype=np.float64)
    m_pvs = len(pv_cols)
    n_reps = len(rep_cols)

    df_complete = float(max(n_reps - 1, 1))
    rows: list[CoefficientRow] = []

    if by is None:
        w_all = np.column_stack([w_full, w_reps])
        stat_cube, stat_names = _compute_group_stat_matrix(y_mat, w_all, q_tuple)
        q_m = stat_cube[0, :, :]  # (M, n_stats)
        u_m = np.empty_like(q_m)
        for pv_i in range(m_pvs):
            u_m[pv_i, :] = replicate_variance(
                stat_cube[0, pv_i, :],
                stat_cube[1:, pv_i, :],
                method=design.variance_method,
                fay_k=design.fay_k,
            )
        pooled, _ = combine_rubin(q_m, u_m, df_complete=df_complete, alpha=alpha)
        for j, name in enumerate(stat_names):
            rows.append(
                CoefficientRow(
                    term=str(name),
                    estimate=float(pooled["q_bar"][j]),
                    se_total=float(pooled["se_total"][j]),
                    se_sampling=float(pooled["se_sampling"][j]),
                    se_imputation=float(pooled["se_imputation"][j]),
                    t_stat=float(pooled["t_stat"][j]),
                    df=float(pooled["df_br"][j]),
                    p_value=float(pooled["p_value"][j]),
                    ci_low=float(pooled["ci_low"][j]),
                    ci_high=float(pooled["ci_high"][j]),
                    fmi=float(pooled["fmi"][j]),
                    rvi=float(pooled["rvi"][j]),
                )
            )
    else:
        groups = list(dict.fromkeys(data[by].tolist()))
        group_cubes: dict[object, np.ndarray] = {}
        stat_names_ref: np.ndarray | None = None
        for grp in groups:
            mask = data[by] == grp
            w_f_dom, w_r_dom = prepare_domain_weights(data, mask, design)
            w_all_dom = np.column_stack([w_f_dom, w_r_dom])
            cube, stat_names = _compute_group_stat_matrix(y_mat, w_all_dom, q_tuple)
            group_cubes[grp] = cube
            stat_names_ref = stat_names

            q_m = cube[0, :, :]
            u_m = np.empty_like(q_m)
            for pv_i in range(m_pvs):
                u_m[pv_i, :] = replicate_variance(
                    cube[0, pv_i, :],
                    cube[1:, pv_i, :],
                    method=design.variance_method,
                    fay_k=design.fay_k,
                )
            pooled, _ = combine_rubin(q_m, u_m, df_complete=df_complete, alpha=alpha)
            for j, sname in enumerate(stat_names):
                rows.append(
                    CoefficientRow(
                        term=f"{by}={grp}:{sname}",
                        estimate=float(pooled["q_bar"][j]),
                        se_total=float(pooled["se_total"][j]),
                        se_sampling=float(pooled["se_sampling"][j]),
                        se_imputation=float(pooled["se_imputation"][j]),
                        t_stat=float(pooled["t_stat"][j]),
                        df=float(pooled["df_br"][j]),
                        p_value=float(pooled["p_value"][j]),
                        ci_low=float(pooled["ci_low"][j]),
                        ci_high=float(pooled["ci_high"][j]),
                        fmi=float(pooled["fmi"][j]),
                        rvi=float(pooled["rvi"][j]),
                    )
                )

        if reference_group is not None:
            if reference_group not in group_cubes:
                raise KeyError(
                    f"reference_group '{reference_group}' not found in column '{by}'."
                )
            ref_cube = group_cubes[reference_group]
            assert stat_names_ref is not None
            for grp in groups:
                if grp == reference_group:
                    continue
                diff_cube = group_cubes[grp] - ref_cube
                q_m = diff_cube[0, :, :]
                u_m = np.empty_like(q_m)
                for pv_i in range(m_pvs):
                    u_m[pv_i, :] = replicate_variance(
                        diff_cube[0, pv_i, :],
                        diff_cube[1:, pv_i, :],
                        method=design.variance_method,
                        fay_k=design.fay_k,
                    )
                pooled, _ = combine_rubin(q_m, u_m, df_complete=df_complete, alpha=alpha)
                for j, sname in enumerate(stat_names_ref):
                    rows.append(
                        CoefficientRow(
                            term=f"diff({grp}-{reference_group}):{sname}",
                            estimate=float(pooled["q_bar"][j]),
                            se_total=float(pooled["se_total"][j]),
                            se_sampling=float(pooled["se_sampling"][j]),
                            se_imputation=float(pooled["se_imputation"][j]),
                            t_stat=float(pooled["t_stat"][j]),
                            df=float(pooled["df_br"][j]),
                            p_value=float(pooled["p_value"][j]),
                            ci_low=float(pooled["ci_low"][j]),
                            ci_high=float(pooled["ci_high"][j]),
                            fmi=float(pooled["fmi"][j]),
                            rvi=float(pooled["rvi"][j]),
                        )
                    )

    n_clusters = (
        int(data[design.psu_col].nunique())
        if design.psu_col and design.psu_col in data.columns
        else None
    )
    return EstimationResult(
        model_type="Weighted PV Descriptive Statistics",
        coefficients=rows,
        n_obs=len(data),
        n_clusters=n_clusters,
        n_pvs=m_pvs,
        n_reps=n_reps,
        variance_method=design.variance_method,
        fay_k=design.fay_k,
        alpha=alpha,
    )
