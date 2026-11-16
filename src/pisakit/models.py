"""
Vectorized regression, within-between decomposition, quantile, two-level
multilevel, and cross-system meta-analytic estimators for plausible-value surveys.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .rubin import combine_rubin
from .types import (
    CoefficientRow,
    DesignSpec,
    EstimationResult,
    MetaPoolingResult,
)
from .weights import (
    compute_senate_weights,
    normalize_weights,
    prepare_domain_weights,
    replicate_covariance,
    replicate_variance,
    validate_survey_frame,
)


def _assemble_design_inputs(
    data: pd.DataFrame,
    outcome_pattern: str,
    predictors: Sequence[str],
    design: DesignSpec,
    system_col: str | None = None,
    add_system_fe: bool = False,
    senate_weights: bool = False,
    senate_target: float = 500.0,
    domain_mask: np.ndarray | pd.Series | None = None,
    extra_required: Sequence[str] | None = None,
) -> tuple[
    np.ndarray,
    list[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int | None,
]:
    """Validate inputs and build design matrix, outcomes, and replicate weights."""
    req = list(predictors)
    if system_col is not None:
        req.append(system_col)
    if extra_required:
        req.extend(extra_required)

    pv_cols, rep_cols = validate_survey_frame(
        data, design, outcome_pattern=outcome_pattern, required_cols=req
    )

    work_df = data
    if senate_weights:
        if system_col is None:
            raise ValueError("system_col must be provided when senate_weights=True.")
        work_df = compute_senate_weights(
            data, design, system_col=system_col, target_sum=senate_target, inplace=False
        )

    if domain_mask is not None:
        w_full, w_reps = prepare_domain_weights(work_df, domain_mask, design)
    else:
        w_full = work_df[design.final_weight].to_numpy(dtype=np.float64)
        w_reps = work_df[rep_cols].to_numpy(dtype=np.float64)

    n_obs = len(work_df)
    x_cols = [np.ones(n_obs, dtype=np.float64)]
    term_names = ["Intercept"]

    for col in predictors:
        arr = work_df[col].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Predictor column '{col}' contains NaN or Inf.")
        x_cols.append(arr)
        term_names.append(col)

    if add_system_fe:
        if system_col is None:
            raise ValueError("system_col must be provided when add_system_fe=True.")
        systems = list(dict.fromkeys(work_df[system_col].tolist()))
        if len(systems) > 1:
            sys_vals = work_df[system_col].to_numpy()
            for sys_code in systems[1:]:
                x_cols.append((sys_vals == sys_code).astype(np.float64))
                term_names.append(f"fe_{sys_code}")

    x_mat = np.column_stack(x_cols)
    y_mat = work_df[pv_cols].to_numpy(dtype=np.float64)

    n_clusters = (
        int(work_df[design.psu_col].nunique())
        if design.psu_col and design.psu_col in work_df.columns
        else None
    )
    if design.stratum_col and design.stratum_col in work_df.columns:
        n_strata = int(work_df[design.stratum_col].nunique())
        df_complete = float(max(n_strata, len(rep_cols) - 1, 1))
    else:
        df_complete = float(max(len(rep_cols) - 1, 1))

    return x_mat, term_names, y_mat, w_full, w_reps, df_complete, n_clusters


def _solve_batched_wls(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    w_all: np.ndarray,
    chunk_size: int = 20000,
) -> np.ndarray:
    """Solve all ``(n_weights x n_pvs)`` weighted regressions via BLAS contractions.

    Computes ``X^T diag(w_r) X`` and ``X^T diag(w_r) Y`` in cache-friendly row
    chunks before solving batched ``(n_weights, p, p)`` linear systems.

    Returns:
        Array ``beta_all`` of shape ``(n_weights, p_dim, n_pvs)``.
    """
    n_obs, p_dim = x_mat.shape
    m_pvs = y_mat.shape[1]
    n_weights = w_all.shape[1]

    xtx_all = np.zeros((n_weights, p_dim, p_dim), dtype=np.float64)
    xty_all = np.zeros((n_weights, p_dim, m_pvs), dtype=np.float64)

    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        xc = x_mat[start:end]
        yc = y_mat[start:end]
        wc = w_all[start:end]
        xx = (xc[:, :, None] * xc[:, None, :]).reshape(end - start, p_dim * p_dim)
        xtx_all += (xx.T @ wc).T.reshape(n_weights, p_dim, p_dim)
        xy = (xc[:, :, None] * yc[:, None, :]).reshape(end - start, p_dim * m_pvs)
        xty_all += (xy.T @ wc).T.reshape(n_weights, p_dim, m_pvs)

    try:
        return np.linalg.solve(xtx_all, xty_all)
    except np.linalg.LinAlgError:
        beta_all = np.empty((n_weights, p_dim, m_pvs), dtype=np.float64)
        for r_idx in range(n_weights):
            beta_all[r_idx], _, _, _ = np.linalg.lstsq(
                xtx_all[r_idx], xty_all[r_idx], rcond=None
            )
        return beta_all


def _pool_parameter_cube(
    beta_all: np.ndarray,
    term_names: list[str],
    design: DesignSpec,
    df_complete: float,
    alpha: float,
) -> tuple[list[CoefficientRow], np.ndarray, np.ndarray]:
    """Combine ``(1 + R, p, M)`` parameter estimates into CoefficientRows."""
    m_pvs = beta_all.shape[2]
    p_dim = beta_all.shape[1]
    q_m = beta_all[0, :, :].T  # (M, p)
    reps_mrp = np.transpose(beta_all[1:, :, :], (2, 0, 1))  # (M, R, p)

    u_m = np.empty((m_pvs, p_dim), dtype=np.float64)
    cov_m = np.empty((m_pvs, p_dim, p_dim), dtype=np.float64)
    for m_idx in range(m_pvs):
        cov_j = replicate_covariance(
            q_m[m_idx],
            reps_mrp[m_idx],
            method=design.variance_method,
            fay_k=design.fay_k,
        )
        cov_m[m_idx] = cov_j
        u_m[m_idx] = np.maximum(np.diag(cov_j), 0.0)

    pooled, _ = combine_rubin(q_m, u_m, df_complete=df_complete, alpha=alpha)
    rows: list[CoefficientRow] = []
    for j, tname in enumerate(term_names):
        rows.append(
            CoefficientRow(
                term=tname,
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
    return rows, q_m, cov_m


def _weighted_r2_across_pvs(
    x_mat: np.ndarray,
    y_mat: np.ndarray,
    w_full: np.ndarray,
    beta_full_mp: np.ndarray,
) -> float:
    """Compute weighted R^2 averaged across all plausible values."""
    w_sum = float(np.sum(w_full))
    y_mean = (w_full @ y_mat) / w_sum  # (M,)
    tss = np.sum(w_full[:, None] * ((y_mat - y_mean[None, :]) ** 2), axis=0)
    preds = x_mat @ beta_full_mp.T  # (N, M)
    rss = np.sum(w_full[:, None] * ((y_mat - preds) ** 2), axis=0)
    r2_m = np.where(tss > 0.0, 1.0 - rss / tss, 0.0)
    return float(np.mean(r2_m))


def fit_pv_ols(
    data: pd.DataFrame,
    outcome_pattern: str,
    predictors: Sequence[str],
    design: DesignSpec,
    system_col: str | None = None,
    add_system_fe: bool = False,
    senate_weights: bool = False,
    senate_target: float = 500.0,
    domain_mask: np.ndarray | pd.Series | None = None,
    alpha: float = 0.05,
) -> EstimationResult:
    """Fit weighted OLS across all plausible values and replicate weights.

    Uses rank-2 BLAS matrix contractions to evaluate all ``M * (1 + R)``
    weighted regressions without Python loops over replicates or PVs.
    """
    x_mat, term_names, y_mat, w_full, w_reps, df_com, n_clusters = _assemble_design_inputs(
        data=data,
        outcome_pattern=outcome_pattern,
        predictors=predictors,
        design=design,
        system_col=system_col,
        add_system_fe=add_system_fe,
        senate_weights=senate_weights,
        senate_target=senate_target,
        domain_mask=domain_mask,
    )
    w_all = np.column_stack([w_full, w_reps])
    beta_all = _solve_batched_wls(x_mat, y_mat, w_all)
    rows, q_m, cov_m = _pool_parameter_cube(beta_all, term_names, design, df_com, alpha)
    r2_mean = _weighted_r2_across_pvs(x_mat, y_mat, w_full, q_m)

    return EstimationResult(
        model_type="Weighted PV OLS Regression",
        coefficients=rows,
        n_obs=int(np.sum(w_full > 0.0)),
        n_clusters=n_clusters,
        n_pvs=y_mat.shape[1],
        n_reps=w_reps.shape[1],
        variance_method=design.variance_method,
        fay_k=design.fay_k,
        alpha=alpha,
        r_squared=r2_mean,
        metadata={
            "q_m": q_m,
            "cov_m": cov_m,
            "term_names": term_names,
            "df_complete": df_com,
        },
    )


def fit_pv_within_between(
    data: pd.DataFrame,
    outcome_pattern: str,
    between_vars: Sequence[str],
    cluster_col: str | None = None,
    controls: Sequence[str] = (),
    design: DesignSpec | None = None,
    system_col: str | None = None,
    add_system_fe: bool = False,
    senate_weights: bool = False,
    alpha: float = 0.05,
) -> EstimationResult:
    """Fit a Mundlak within-between decomposition across PVs and replicates.

    Decomposes each variable ``v`` in ``between_vars`` into its weighted cluster
    mean ``{v}_between`` and student within-cluster deviation ``{v}_within``,
    and reports the exact contextual difference ``{v}_contextual = beta_between - beta_within``
    with full Rubin + Fay BRR variance.
    """
    if design is None:
        design = DesignSpec.from_preset("pisa")
    grp_col = cluster_col or design.psu_col
    if grp_col is None or grp_col not in data.columns:
        raise KeyError("A valid cluster_col (or design.psu_col) is required for within-between decomposition.")
    if not between_vars:
        raise ValueError("between_vars must contain at least one predictor.")

    req = list(between_vars) + list(controls) + [grp_col]
    if system_col:
        req.append(system_col)
    validate_survey_frame(data, design, outcome_pattern=outcome_pattern, required_cols=req)

    work = data.copy()
    w = work[design.final_weight].to_numpy(dtype=np.float64)
    _, inv = np.unique(work[grp_col].to_numpy(), return_inverse=True)
    n_groups = int(inv.max()) + 1
    w_grp_sums = np.bincount(inv, weights=w, minlength=n_groups)
    if np.any(w_grp_sums <= 0.0):
        raise ValueError("Every cluster must have strictly positive total weight.")

    decomp_predictors: list[str] = []
    pair_indices: list[tuple[str, int, int]] = []
    for var in between_vars:
        vals = work[var].to_numpy(dtype=np.float64)
        grp_means = np.bincount(inv, weights=w * vals, minlength=n_groups) / w_grp_sums
        mean_col = f"{var}_between"
        within_col = f"{var}_within"
        work[mean_col] = grp_means[inv]
        work[within_col] = vals - work[mean_col]
        b_idx = 1 + len(decomp_predictors)
        w_idx = 1 + len(decomp_predictors) + 1
        decomp_predictors.extend([mean_col, within_col])
        pair_indices.append((var, b_idx, w_idx))

    decomp_predictors.extend(controls)

    x_mat, term_names, y_mat, w_full, w_reps, df_com, n_clusters = _assemble_design_inputs(
        data=work,
        outcome_pattern=outcome_pattern,
        predictors=decomp_predictors,
        design=design,
        system_col=system_col,
        add_system_fe=add_system_fe,
        senate_weights=senate_weights,
    )
    w_all = np.column_stack([w_full, w_reps])
    beta_all = _solve_batched_wls(x_mat, y_mat, w_all)

    # Append contextual contrasts beta_between - beta_within across all replicates & PVs
    ctx_blocks = []
    ctx_names = []
    for var, b_idx, w_idx in pair_indices:
        ctx_blocks.append(beta_all[:, b_idx : b_idx + 1, :] - beta_all[:, w_idx : w_idx + 1, :])
        ctx_names.append(f"{var}_contextual")
    beta_aug = np.concatenate([beta_all] + ctx_blocks, axis=1)
    aug_names = term_names + ctx_names

    rows, q_m, cov_m = _pool_parameter_cube(beta_aug, aug_names, design, df_com, alpha)
    r2_mean = _weighted_r2_across_pvs(x_mat, y_mat, w_full, q_m[:, : len(term_names)])

    return EstimationResult(
        model_type="Weighted PV Within-Between (Mundlak) Decomposition",
        coefficients=rows,
        n_obs=int(np.sum(w_full > 0.0)),
        n_clusters=n_clusters,
        n_pvs=y_mat.shape[1],
        n_reps=w_reps.shape[1],
        variance_method=design.variance_method,
        fay_k=design.fay_k,
        alpha=alpha,
        r_squared=r2_mean,
        metadata={"q_m": q_m, "cov_m": cov_m, "term_names": aug_names},
    )


def _irls_weighted_quantile(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    weights: np.ndarray,
    tau: float,
    beta_init: np.ndarray,
    eps: float,
    max_iter: int = 35,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve single weighted quantile regression via smoothed IRLS."""
    beta = beta_init.copy()
    for _ in range(max_iter):
        resid = y_vec - x_mat @ beta
        asymm = np.where(resid >= 0.0, tau, 1.0 - tau)
        denom = np.maximum(np.abs(resid), eps)
        w_eff = weights * (asymm / denom)
        xtw = x_mat.T * w_eff[None, :]
        lhs = xtw @ x_mat
        rhs = xtw @ y_vec
        try:
            beta_new = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            beta_new, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            return beta_new
        beta = beta_new
    return beta


def fit_pv_quantile(
    data: pd.DataFrame,
    outcome_pattern: str,
    predictors: Sequence[str],
    design: DesignSpec,
    tau: float = 0.5,
    system_col: str | None = None,
    add_system_fe: bool = False,
    senate_weights: bool = False,
    max_iter: int = 35,
    alpha: float = 0.05,
) -> EstimationResult:
    """Fit survey-weighted quantile regression across PVs and replicate weights.

    Warm-starts IRLS from the vectorized weighted OLS solution so every PV and
    replicate converges smoothly across quantiles ``tau in (0, 1)``.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"Quantile level tau must satisfy 0 < tau < 1, got {tau}")

    x_mat, term_names, y_mat, w_full, w_reps, df_com, n_clusters = _assemble_design_inputs(
        data=data,
        outcome_pattern=outcome_pattern,
        predictors=predictors,
        design=design,
        system_col=system_col,
        add_system_fe=add_system_fe,
        senate_weights=senate_weights,
    )
    w_all = np.column_stack([w_full, w_reps])
    ols_init = _solve_batched_wls(x_mat, y_mat, w_full[:, None])[0]  # (p, M)

    n_weights = w_all.shape[1]
    p_dim = x_mat.shape[1]
    m_pvs = y_mat.shape[1]
    beta_all = np.empty((n_weights, p_dim, m_pvs), dtype=np.float64)

    for m_idx in range(m_pvs):
        y_vec = y_mat[:, m_idx]
        y_scale = max(float(np.std(y_vec)), 1.0)
        eps = 1e-4 * y_scale
        beta_full_q = _irls_weighted_quantile(
            x_mat, y_vec, w_all[:, 0], tau=tau, beta_init=ols_init[:, m_idx], eps=eps, max_iter=max_iter
        )
        beta_all[0, :, m_idx] = beta_full_q
        for r_idx in range(1, n_weights):
            beta_all[r_idx, :, m_idx] = _irls_weighted_quantile(
                x_mat, y_vec, w_all[:, r_idx], tau=tau, beta_init=beta_full_q, eps=eps, max_iter=max_iter
            )

    rows, q_m, cov_m = _pool_parameter_cube(beta_all, term_names, design, df_com, alpha)
    return EstimationResult(
        model_type=f"Weighted PV Quantile Regression (tau={tau:.2f})",
        coefficients=rows,
        n_obs=int(np.sum(w_full > 0.0)),
        n_clusters=n_clusters,
        n_pvs=m_pvs,
        n_reps=w_reps.shape[1],
        variance_method=design.variance_method,
        fay_k=design.fay_k,
        alpha=alpha,
        fit_stats={"tau": float(tau)},
        metadata={"q_m": q_m, "cov_m": cov_m, "term_names": term_names},
    )


def _solve_random_intercept_reml(
    x_mat: np.ndarray,
    y_vec: np.ndarray,
    w_vec: np.ndarray,
    group_inv: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, float, float, float]:
    """Fit weighted 2-level random-intercept REML via Woodbury cluster updates."""
    p_dim = x_mat.shape[1]
    w_sum_eff = float(np.sum(w_vec))
    df_resid = max(w_sum_eff - p_dim, 1.0)

    # Precompute sufficient statistics per cluster once for fast 1D profile REML
    s_w = np.bincount(group_inv, weights=w_vec, minlength=n_groups)
    s_x = np.zeros((n_groups, p_dim), dtype=np.float64)
    s_y = np.bincount(group_inv, weights=w_vec * y_vec, minlength=n_groups)
    s_xx = np.zeros((n_groups, p_dim, p_dim), dtype=np.float64)
    s_xy = np.zeros((n_groups, p_dim), dtype=np.float64)
    s_yy = np.bincount(group_inv, weights=w_vec * y_vec * y_vec, minlength=n_groups)

    for j_col in range(p_dim):
        wx_j = w_vec * x_mat[:, j_col]
        s_x[:, j_col] = np.bincount(group_inv, weights=wx_j, minlength=n_groups)
        s_xy[:, j_col] = np.bincount(group_inv, weights=wx_j * y_vec, minlength=n_groups)
        for k_col in range(j_col, p_dim):
            prod = np.bincount(
                group_inv, weights=wx_j * x_mat[:, k_col], minlength=n_groups
            )
            s_xx[:, j_col, k_col] = prod
            if k_col != j_col:
                s_xx[:, k_col, j_col] = prod

    total_xx = np.sum(s_xx, axis=0)
    total_xy = np.sum(s_xy, axis=0)
    total_yy = float(np.sum(s_yy))

    def _eval_at_ratio(lam: float) -> tuple[float, np.ndarray, float]:
        alpha_j = lam / (1.0 + lam * s_w)
        # Weighted cross-products via rank-1 Woodbury subtraction
        adj_xx = total_xx - (s_x.T * alpha_j[None, :]) @ s_x
        adj_xy = total_xy - np.sum((alpha_j * s_y)[:, None] * s_x, axis=0)
        try:
            beta = np.linalg.solve(adj_xx, adj_xy)
        except np.linalg.LinAlgError:
            beta, _, _, _ = np.linalg.lstsq(adj_xx, adj_xy, rcond=None)
        sign_det, logdet_xx = np.linalg.slogdet(adj_xx)
        if sign_det <= 0:
            logdet_xx = 50.0

        # Exact generalized RSS = sum_j [e_j^T W_j e_j - alpha_j (e_j^T w_j)^2]
        ew_j = s_y - s_x @ beta
        raw_ewe = (
            total_yy
            - 2.0 * float(beta @ total_xy)
            + float(beta @ total_xx @ beta)
        )
        rss = max(raw_ewe - float(np.sum(alpha_j * (ew_j ** 2))), 1e-12)
        sigma2 = rss / df_resid
        logdet_v = float(np.sum(np.log(np.maximum(1.0 + lam * s_w, 1e-12))))
        reml_obj = logdet_v + float(logdet_xx) + df_resid * np.log(max(sigma2, 1e-12))
        return reml_obj, beta, sigma2

    opt = optimize.minimize_scalar(
        lambda log_lam: _eval_at_ratio(float(np.exp(log_lam)))[0],
        bounds=(-12.0, 6.0),
        method="bounded",
    )
    lam_hat = float(np.exp(opt.x))
    _, beta_hat, sigma2_hat = _eval_at_ratio(lam_hat)
    tau2_hat = lam_hat * sigma2_hat
    icc_hat = tau2_hat / max(tau2_hat + sigma2_hat, 1e-12)
    return beta_hat, tau2_hat, sigma2_hat, icc_hat


def fit_pv_multilevel(
    data: pd.DataFrame,
    outcome_pattern: str,
    predictors: Sequence[str],
    cluster_col: str | None = None,
    design: DesignSpec | None = None,
    weight_scaling: str = "mean_one",
    system_col: str | None = None,
    add_system_fe: bool = False,
    alpha: float = 0.05,
) -> EstimationResult:
    """Fit a weighted two-level random-intercept model across PVs and replicates.

    Uses Asparouhov (2006) within-cluster student weight scaling and exact
    cluster-Woodbury weighted REML profile likelihood estimation for fixed effects,
    school-level variance ``tau2_between``, residual variance ``sigma2_within``,
    and the intraclass correlation ``ICC``.
    """
    if design is None:
        design = DesignSpec.from_preset("pisa")
    grp_col = cluster_col or design.psu_col
    if grp_col is None or grp_col not in data.columns:
        raise KeyError("A valid cluster_col (or design.psu_col) is required for multilevel modeling.")

    x_mat, term_names, y_mat, w_full, w_reps, _, n_clusters = _assemble_design_inputs(
        data=data,
        outcome_pattern=outcome_pattern,
        predictors=predictors,
        design=design,
        system_col=system_col,
        add_system_fe=add_system_fe,
        extra_required=[grp_col],
    )
    _, group_inv = np.unique(data[grp_col].to_numpy(), return_inverse=True)
    n_groups = int(group_inv.max()) + 1
    df_complete = float(max(n_groups - x_mat.shape[1], 1))

    # Build normalized weight columns across full sample and all replicates
    work_w = data[[design.final_weight]].copy()
    w_norm_cols: list[np.ndarray] = [
        normalize_weights(work_w.assign(_g=data[grp_col]), design.final_weight, "_g", method=weight_scaling)
    ]
    _, rep_cols = validate_survey_frame(data, design)
    for rcol in rep_cols:
        w_norm_cols.append(
            normalize_weights(
                pd.DataFrame({rcol: data[rcol], "_g": data[grp_col]}),
                rcol,
                "_g",
                method=weight_scaling,
            )
        )
    w_norm_all = np.column_stack(w_norm_cols)  # (N, 1 + R)

    n_weights = w_norm_all.shape[1]
    p_dim = x_mat.shape[1]
    m_pvs = y_mat.shape[1]

    # Store [beta..., tau2_school, sigma2_student, icc] uniformly in parameter cube
    aug_names = term_names + ["tau2_school", "sigma2_student", "icc"]
    beta_aug = np.empty((n_weights, p_dim + 3, m_pvs), dtype=np.float64)

    for m_idx in range(m_pvs):
        y_vec = y_mat[:, m_idx]
        for w_idx in range(n_weights):
            b_hat, tau2, sig2, icc = _solve_random_intercept_reml(
                x_mat, y_vec, w_norm_all[:, w_idx], group_inv, n_groups
            )
            beta_aug[w_idx, :p_dim, m_idx] = b_hat
            beta_aug[w_idx, p_dim, m_idx] = tau2
            beta_aug[w_idx, p_dim + 1, m_idx] = sig2
            beta_aug[w_idx, p_dim + 2, m_idx] = icc

    rows, q_m, cov_m = _pool_parameter_cube(beta_aug, aug_names, design, df_complete, alpha)
    fixed_rows = rows[:p_dim]
    var_map = {r.term: r.estimate for r in rows[p_dim:]}

    return EstimationResult(
        model_type="Weighted Two-Level Random-Intercept Model (REML + PV/BRR)",
        coefficients=fixed_rows,
        n_obs=len(data),
        n_clusters=n_groups if n_clusters is None else n_clusters,
        n_pvs=m_pvs,
        n_reps=w_reps.shape[1],
        variance_method=design.variance_method,
        fay_k=design.fay_k,
        alpha=alpha,
        fit_stats=var_map,
        metadata={"q_m": q_m[:, :p_dim], "cov_m": cov_m[:, :p_dim, :p_dim], "term_names": term_names},
    )


def pool_cross_system(
    data: pd.DataFrame,
    outcome_pattern: str,
    predictors: Sequence[str],
    term: str,
    system_col: str = "CNT",
    design: DesignSpec | None = None,
    estimator: Callable[..., EstimationResult] | None = None,
    estimator_kwargs: dict[str, Any] | None = None,
    alpha: float = 0.05,
) -> MetaPoolingResult:
    """Estimate a focal coefficient per education system and pool across systems.

    Performs inverse-variance fixed-effect and DerSimonian-Laird (1986)
    random-effects meta-analysis, reporting Cochran's Q, ``tau^2``, and ``I^2``.
    """
    if design is None:
        design = DesignSpec.from_preset("pisa")
    if system_col not in data.columns:
        raise KeyError(f"System column '{system_col}' not found in DataFrame.")

    fit_fn = fit_pv_ols if estimator is None else estimator
    extra_kw = dict(estimator_kwargs or {})

    systems = list(dict.fromkeys(data[system_col].tolist()))
    if len(systems) < 2:
        raise ValueError(f"At least 2 distinct systems required for meta pooling, got {len(systems)}.")

    sys_records: list[dict[str, Any]] = []
    for sys_code in systems:
        sub = data[data[system_col] == sys_code]
        res = fit_fn(
            data=sub,
            outcome_pattern=outcome_pattern,
            predictors=predictors,
            design=design,
            alpha=alpha,
            **extra_kw,
        )
        row = res.get_term(term)
        sys_records.append(
            {
                "system": str(sys_code),
                "n_obs": res.n_obs,
                "estimate": row.estimate,
                "se": row.se_total,
                "df": row.df,
                "p_value": row.p_value,
                "ci_low": row.ci_low,
                "ci_high": row.ci_high,
                "fmi": row.fmi,
            }
        )

    sys_df = pd.DataFrame(sys_records)
    est = sys_df["estimate"].to_numpy(dtype=np.float64)
    vi = np.maximum(sys_df["se"].to_numpy(dtype=np.float64) ** 2, 1e-18)
    w_fe = 1.0 / vi
    sum_w = float(np.sum(w_fe))

    pooled_fe = float(np.sum(w_fe * est) / sum_w)
    se_fe = float(np.sqrt(1.0 / sum_w))
    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_fe = (pooled_fe - z_crit * se_fe, pooled_fe + z_crit * se_fe)
    p_fe = float(2.0 * stats.norm.sf(abs(pooled_fe / se_fe)))

    q_stat = float(np.sum(w_fe * ((est - pooled_fe) ** 2)))
    q_df = len(systems) - 1
    q_pvalue = float(stats.chi2.sf(q_stat, df=q_df))

    c_denom = sum_w - float(np.sum(w_fe * w_fe)) / sum_w
    tau2 = max((q_stat - q_df) / c_denom, 0.0) if c_denom > 0.0 else 0.0
    i2 = max((q_stat - q_df) / q_stat, 0.0) if q_stat > 0.0 else 0.0

    w_re = 1.0 / (vi + tau2)
    sum_w_re = float(np.sum(w_re))
    pooled_re = float(np.sum(w_re * est) / sum_w_re)
    se_re = float(np.sqrt(1.0 / sum_w_re))
    ci_re = (pooled_re - z_crit * se_re, pooled_re + z_crit * se_re)
    p_re = float(2.0 * stats.norm.sf(abs(pooled_re / se_re)))

    sys_df["weight_fe_pct"] = 100.0 * w_fe / sum_w
    sys_df["weight_re_pct"] = 100.0 * w_re / sum_w_re

    return MetaPoolingResult(
        term=term,
        systems=[str(s) for s in systems],
        pooled_fe=pooled_fe,
        se_fe=se_fe,
        ci_fe=ci_fe,
        p_fe=p_fe,
        pooled_re=pooled_re,
        se_re=se_re,
        ci_re=ci_re,
        p_re=p_re,
        tau2=tau2,
        i2=i2,
        q_stat=q_stat,
        q_df=q_df,
        q_pvalue=q_pvalue,
        system_table=sys_df,
    )
