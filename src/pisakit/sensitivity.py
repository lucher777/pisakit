"""
Cinelli-Hazlett (2020) omitted-variable bias (OVB) sensitivity analysis coupled
with survey Rubin + Barnard-Rubin + Fay BRR variance and degrees of freedom.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .models import fit_pv_ols
from .types import DesignSpec, EstimationResult, SensitivitySummary


def partial_r2_from_t(t_stat: float, df: float) -> float:
    """Convert a regression t-statistic and degrees of freedom to partial R^2."""
    if df <= 0.0:
        raise ValueError(f"Degrees of freedom must be positive, got {df}")
    t2 = float(t_stat) ** 2
    return float(t2 / (t2 + df))


def compute_robustness_values(
    t_stat: float,
    df: float,
    q: float = 1.0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Compute Cinelli-Hazlett robustness values ``RV_q`` and ``RV_{q, alpha}``.

    Uses exact Student-t critical value ``t_crit`` evaluated at the survey
    Barnard-Rubin degrees of freedom ``df`` rather than asymptotic normal quantiles.

    Returns:
        ``(r2_yd_x, rv_q, rv_qa)``.
    """
    if not (0.0 < q <= 1.0):
        raise ValueError(f"Proportion reduction q must be in (0, 1], got {q}")
    if df <= 0.0:
        raise ValueError(f"Degrees of freedom must be positive, got {df}")

    r2_yd_x = partial_r2_from_t(t_stat, df)
    abs_t = abs(float(t_stat))

    def _rv_from_f(f_val: float) -> float:
        return 0.5 * (np.sqrt((f_val ** 4) + 4.0 * (f_val ** 2)) - (f_val ** 2))

    fq = q * abs_t / np.sqrt(df)
    rv_q = float(np.clip(_rv_from_f(fq), 0.0, 1.0))

    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df)) if np.isfinite(df) else float(
        stats.norm.ppf(1.0 - alpha / 2.0)
    )
    f_alpha = t_crit / np.sqrt(df) if np.isfinite(df) else 0.0
    f_qa = fq - f_alpha
    if f_qa <= 0.0:
        rv_qa = 0.0
    else:
        rv_qa = float(np.clip(_rv_from_f(f_qa), 0.0, 1.0))

    return r2_yd_x, rv_q, rv_qa


def adjusted_estimate(
    estimate: float,
    se: float,
    df: float,
    r2_dz_x: float | np.ndarray,
    r2_yz_dx: float | np.ndarray,
    direction: float = -1.0,
    alpha: float = 0.05,
) -> dict[str, np.ndarray | float]:
    """Compute confounder-adjusted point estimate, SE, and t-statistic.

    Follows Theorems 2 and 3 of Cinelli & Hazlett (2020):
        ``|bias| = se * sqrt(df * (r2_yz_dx * r2_dz_x) / (1 - r2_dz_x))``
        ``se_adj = se * sqrt((df / (df - 1)) * (1 - r2_yz_dx) / (1 - r2_dz_x))``
    """
    dz = np.asarray(r2_dz_x, dtype=np.float64)
    yz = np.asarray(r2_yz_dx, dtype=np.float64)
    if np.any(dz < 0.0) or np.any(dz >= 1.0):
        raise ValueError("r2_dz_x must lie strictly in [0, 1).")
    if np.any(yz < 0.0) or np.any(yz > 1.0):
        raise ValueError("r2_yz_dx must lie in [0, 1].")
    if df <= 1.0:
        raise ValueError(f"Degrees of freedom must exceed 1, got {df}")

    bf = np.sqrt((yz * dz) / np.maximum(1.0 - dz, 1e-12))
    bias_mag = se * np.sqrt(df) * bf
    sign = np.sign(estimate) if estimate != 0.0 else 1.0
    adj_est = estimate + direction * sign * bias_mag

    df_adj = df - 1.0
    adj_se = se * np.sqrt((df / df_adj) * np.maximum(1.0 - yz, 0.0) / np.maximum(1.0 - dz, 1e-12))
    adj_t = np.where(adj_se > 0.0, adj_est / adj_se, 0.0)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df_adj)) if np.isfinite(df_adj) else float(
        stats.norm.ppf(1.0 - alpha / 2.0)
    )
    ci_low = adj_est - t_crit * adj_se
    ci_high = adj_est + t_crit * adj_se

    return {
        "adjusted_estimate": adj_est,
        "adjusted_se": adj_se,
        "adjusted_t": adj_t,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "bias": bias_mag,
    }


def benchmark_confounder_bounds(
    data: pd.DataFrame,
    outcome_pattern: str,
    treatment: str,
    controls: Sequence[str],
    benchmarks: Sequence[str],
    design: DesignSpec,
    base_result: EstimationResult | None = None,
    multipliers: Sequence[float] = (1.0, 2.0, 3.0),
    system_col: str | None = None,
    add_system_fe: bool = False,
    senate_weights: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute informal partial-R^2 confounder bounds using observed covariates.

    For each benchmark covariate ``z`` in ``controls``, evaluates its survey-weighted
    partial ``R^2`` with the outcome (controlling for treatment + other controls) and
    with the treatment (controlling for other controls), then scales by
    ``multipliers`` (e.g., ``1x, 2x, 3x`` strength of ESCS).
    """
    all_predictors = list(dict.fromkeys([treatment] + list(controls)))
    if base_result is None:
        base_result = fit_pv_ols(
            data=data,
            outcome_pattern=outcome_pattern,
            predictors=all_predictors,
            design=design,
            system_col=system_col,
            add_system_fe=add_system_fe,
            senate_weights=senate_weights,
            alpha=alpha,
        )

    treat_row = base_result.get_term(treatment)
    est = treat_row.estimate
    se = treat_row.se_total
    df_treat = treat_row.df

    treat_design = DesignSpec(
        name=design.name,
        final_weight=design.final_weight,
        rep_prefix=design.rep_prefix,
        n_reps=design.n_reps,
        variance_method=design.variance_method,
        fay_k=design.fay_k,
        n_pvs=1,
        stratum_col=design.stratum_col,
        psu_col=design.psu_col,
        index_start=design.index_start,
    )

    records: list[dict[str, float | str]] = []
    for bench in benchmarks:
        if bench == treatment:
            raise ValueError(f"Benchmark '{bench}' cannot be the same column as treatment.")
        if bench not in all_predictors:
            raise KeyError(f"Benchmark '{bench}' must be included in controls.")

        bench_y_row = base_result.get_term(bench)
        r2_yz_dx_base = partial_r2_from_t(bench_y_row.t_stat, bench_y_row.df)

        treat_controls = [c for c in controls if c != bench] + [bench]
        treat_reg = fit_pv_ols(
            data=data,
            outcome_pattern=treatment,
            predictors=treat_controls,
            design=treat_design,
            system_col=system_col,
            add_system_fe=add_system_fe,
            senate_weights=senate_weights,
            alpha=alpha,
        )
        bench_d_row = treat_reg.get_term(bench)
        r2_dz_x_base = partial_r2_from_t(bench_d_row.t_stat, bench_d_row.df)

        for mult in multipliers:
            r2_dz_bound = float(np.clip(mult * r2_dz_x_base, 0.0, 0.999))
            r2_yz_bound = float(np.clip(mult * r2_yz_dx_base, 0.0, 0.999))
            adj = adjusted_estimate(
                estimate=est,
                se=se,
                df=df_treat,
                r2_dz_x=r2_dz_bound,
                r2_yz_dx=r2_yz_bound,
                direction=-1.0,
                alpha=alpha,
            )
            records.append(
                {
                    "benchmark": bench,
                    "multiplier": float(mult),
                    "r2_dz_x": r2_dz_bound,
                    "r2_yz_dx": r2_yz_bound,
                    "adjusted_estimate": float(adj["adjusted_estimate"]),
                    "adjusted_se": float(adj["adjusted_se"]),
                    "adjusted_t": float(adj["adjusted_t"]),
                    "ci_low": float(adj["ci_low"]),
                    "ci_high": float(adj["ci_high"]),
                }
            )
    return pd.DataFrame(records)


def sensitivity_contour_grid(
    estimate: float,
    se: float,
    df: float,
    max_r2_dz: float = 0.25,
    max_r2_yz: float = 0.25,
    n_grid: int = 60,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Generate a 2D mesh grid for Cinelli-Hazlett estimate and t-value contours."""
    dz_axis = np.linspace(0.0, max_r2_dz, n_grid)
    yz_axis = np.linspace(0.0, max_r2_yz, n_grid)
    dz_grid, yz_grid = np.meshgrid(dz_axis, yz_axis)
    adj = adjusted_estimate(
        estimate=estimate,
        se=se,
        df=df,
        r2_dz_x=dz_grid,
        r2_yz_dx=yz_grid,
        direction=-1.0,
        alpha=alpha,
    )
    return {
        "r2_dz_grid": dz_grid,
        "r2_yz_grid": yz_grid,
        "adjusted_estimate": np.asarray(adj["adjusted_estimate"], dtype=np.float64),
        "adjusted_t": np.asarray(adj["adjusted_t"], dtype=np.float64),
    }


def sensitivity_analysis(
    data: pd.DataFrame,
    outcome_pattern: str,
    treatment: str,
    controls: Sequence[str],
    design: DesignSpec | None = None,
    benchmarks: Sequence[str] = (),
    multipliers: Sequence[float] = (1.0, 2.0, 3.0),
    q_threshold: float = 1.0,
    system_col: str | None = None,
    add_system_fe: bool = False,
    senate_weights: bool = False,
    alpha: float = 0.05,
) -> SensitivitySummary:
    """Run complete OVB sensitivity analysis from raw PV/BRR survey data."""
    if design is None:
        design = DesignSpec.from_preset("pisa")

    all_predictors = list(dict.fromkeys([treatment] + list(controls)))
    base_res = fit_pv_ols(
        data=data,
        outcome_pattern=outcome_pattern,
        predictors=all_predictors,
        design=design,
        system_col=system_col,
        add_system_fe=add_system_fe,
        senate_weights=senate_weights,
        alpha=alpha,
    )
    treat_row = base_res.get_term(treatment)
    r2_yd_x, rv_q, rv_qa = compute_robustness_values(
        t_stat=treat_row.t_stat,
        df=treat_row.df,
        q=q_threshold,
        alpha=alpha,
    )
    t_crit = (
        float(stats.t.ppf(1.0 - alpha / 2.0, df=treat_row.df))
        if np.isfinite(treat_row.df)
        else float(stats.norm.ppf(1.0 - alpha / 2.0))
    )

    bench_df = pd.DataFrame()
    if benchmarks:
        bench_df = benchmark_confounder_bounds(
            data=data,
            outcome_pattern=outcome_pattern,
            treatment=treatment,
            controls=controls,
            benchmarks=benchmarks,
            design=design,
            base_result=base_res,
            multipliers=multipliers,
            system_col=system_col,
            add_system_fe=add_system_fe,
            senate_weights=senate_weights,
            alpha=alpha,
        )

    return SensitivitySummary(
        term=treatment,
        estimate=treat_row.estimate,
        se=treat_row.se_total,
        df=treat_row.df,
        t_stat=treat_row.t_stat,
        t_crit=t_crit,
        q_threshold=q_threshold,
        alpha=alpha,
        r2_yd_x=r2_yd_x,
        rv_q=rv_q,
        rv_qa=rv_qa,
        benchmark_bounds=bench_df,
    )
