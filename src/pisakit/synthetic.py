"""
Synthetic international large-scale assessment (LSA) survey data generator
and Monte Carlo coverage benchmarking engine.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd


_PALEY_PRIMES_3_MOD_4 = (3, 7, 11, 19, 23, 31, 43, 47, 59, 67, 71, 79, 83)


def _paley_hadamard_base(prime: int) -> np.ndarray:
    """Construct an exact Paley Type I Hadamard matrix of order ``prime + 1``.

    Requires ``prime`` to be an odd prime satisfying ``prime % 4 == 3``.
    """
    residues = {(x * x) % prime for x in range(1, prime)}
    q_mat = np.zeros((prime, prime), dtype=np.float64)
    for i in range(prime):
        for j in range(prime):
            diff = (i - j) % prime
            if diff != 0:
                q_mat[i, j] = 1.0 if diff in residues else -1.0
    order = prime + 1
    h_mat = np.ones((order, order), dtype=np.float64)
    h_mat[1:, 1:] = q_mat - np.eye(prime, dtype=np.float64)
    return h_mat


def _try_build_hadamard(order: int) -> np.ndarray | None:
    """Recursively attempt Hadamard construction via Sylvester and Paley bases."""
    if order == 1:
        return np.ones((1, 1), dtype=np.float64)
    if order == 2:
        return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    if (order & (order - 1)) == 0:
        half = _try_build_hadamard(order // 2)
        if half is not None:
            return np.kron(np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64), half)

    for prime in _PALEY_PRIMES_3_MOD_4:
        if order == prime + 1:
            return _paley_hadamard_base(prime)

    for prime in reversed(_PALEY_PRIMES_3_MOD_4):
        base_order = prime + 1
        if order % base_order == 0 and order > base_order:
            sub = _try_build_hadamard(order // base_order)
            if sub is not None:
                return np.kron(sub, _paley_hadamard_base(prime))

    if order % 2 == 0:
        sub = _try_build_hadamard(order // 2)
        if sub is not None:
            return np.kron(np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64), sub)
    return None


def build_hadamard_matrix(order: int) -> np.ndarray:
    """Build an orthogonal Hadamard matrix of size ``(order, order)``.

    Supports powers of two via Sylvester's recursion, Paley Type I bases
    (e.g., orders 12, 20, 24, 32, 40, 48, 80, ...), and Kronecker factors.

    Raises:
        ValueError: If ``order`` cannot be factored from supported bases.
    """
    if order < 1:
        raise ValueError(f"Hadamard order must be >= 1, got {order}")
    mat = _try_build_hadamard(order)
    if mat is not None:
        return mat
    raise ValueError(
        f"Cannot construct Hadamard matrix of order {order}. "
        "Choose 1, 2, a power of 2, or a multiple of 4*p' (e.g., 12, 20, 40, 80)."
    )


def generate_fay_brr_matrix(
    n_strata: int,
    n_reps: int = 80,
    fay_k: float = 0.5,
) -> np.ndarray:
    """Generate replicate multiplier factors for paired variance strata.

    Returns an array of shape ``(n_strata, 2, n_reps)`` where entry ``[s, psu, r]``
    is the Fay BRR weight multiplier for ``psu in {0, 1}`` in stratum ``s`` and
    replicate ``r``. Strictly uses non-constant orthogonal rows of ``H_{n_reps}``
    so cross-stratum replicate covariances vanish when ``n_strata <= n_reps - 1``.
    """
    if n_strata < 1:
        raise ValueError(f"n_strata must be >= 1, got {n_strata}")
    if not (0.0 <= fay_k < 1.0):
        raise ValueError(f"fay_k must be in [0, 1), got {fay_k}")

    h_mat = build_hadamard_matrix(n_reps)
    # Row 0 is all +1; use rows 1..n_reps-1 whose replicate sums equal zero.
    orthogonal_rows = h_mat[1:, :]
    n_avail = orthogonal_rows.shape[0]
    row_indices = np.arange(n_strata) % n_avail
    signs = orthogonal_rows[row_indices, :]  # (n_strata, n_reps)

    step = 1.0 - fay_k
    mult = np.empty((n_strata, 2, n_reps), dtype=np.float64)
    mult[:, 0, :] = 1.0 + step * signs
    mult[:, 1, :] = 1.0 - step * signs
    return mult


def simulate_lsa_survey(
    n_systems: int = 4,
    n_strata_per_system: int = 15,
    students_per_school: int = 20,
    n_pvs: int = 10,
    n_reps: int = 80,
    fay_k: float = 0.5,
    pv_sd: float = 28.0,
    school_icc: float = 0.25,
    true_intercept: float = 500.0,
    true_escs: float = 32.0,
    true_female: float = -6.0,
    true_treatment: float = 18.0,
    true_school_escs: float = 20.0,
    unmeasured_confounding: float = 0.0,
    system_heterogeneity_sd: float = 4.0,
    seed: int = 2026,
) -> pd.DataFrame:
    """Simulate a hierarchical multi-system LSA survey dataset.

    Each system contains ``n_strata_per_system`` explicit variance strata, with
    exactly 2 PSUs (schools) per stratum and ``students_per_school`` students per
    school. Generates unequal school selection probabilities, student final
    weights ``W_FSTUWT``, ``n_reps`` Fay BRR weights ``W_FSTURWT1..R``, and
    ``n_pvs`` plausible values ``PV1MATH..PVMMATH`` drawn around student latent
    proficiency conditional on observable and unobservable factors.
    """
    rng = np.random.default_rng(seed)
    default_codes = ["SGP", "FIN", "JPN", "DEU", "USA", "BRA", "KOR", "CHL", "NOR", "POL"]
    systems = [
        default_codes[i] if i < len(default_codes) else f"SYS{i + 1:02d}"
        for i in range(n_systems)
    ]

    total_strata = n_systems * n_strata_per_system
    hadamard_order = n_reps
    brr_mult = generate_fay_brr_matrix(n_strata=total_strata, n_reps=hadamard_order, fay_k=fay_k)

    sigma_total = 85.0
    sigma_u = sigma_total * np.sqrt(school_icc)
    sigma_e = sigma_total * np.sqrt(max(1.0 - school_icc, 1e-6))

    records: list[dict[str, Any]] = []
    global_stratum = 0
    school_counter = 1000
    student_counter = 100000

    # Accumulate arrays per row so replicate weight matrix can be attached cleanly
    rep_weight_rows: list[np.ndarray] = []

    for sys_idx, sys_code in enumerate(systems):
        sys_treat_shift = rng.normal(0.0, system_heterogeneity_sd)
        sys_intercept_shift = rng.normal(0.0, 22.0)
        beta_treat_sys = true_treatment + sys_treat_shift

        for local_s in range(n_strata_per_system):
            stratum_id = f"{sys_code}_STR{local_s + 1:02d}"
            # Stratum socioeconomic composition + systematic weighting
            stratum_escs = rng.normal(0.0, 0.45)

            for psu_idx in range(2):
                school_counter += 1
                school_id = school_counter
                # Varying school selection weight correlated gently with stratum ESCS
                school_weight = float(
                    np.clip(
                        rng.lognormal(mean=2.3 - 0.15 * stratum_escs, sigma=0.35),
                        3.0,
                        45.0,
                    )
                )
                u_school = rng.normal(0.0, sigma_u)
                school_escs = stratum_escs + rng.normal(0.0, 0.35)
                rep_mult = brr_mult[global_stratum, psu_idx, :]  # shape (n_reps,)

                for _ in range(students_per_school):
                    student_counter += 1
                    escs = school_escs + rng.normal(0.0, 0.85)
                    female = int(rng.random() < 0.50)
                    u_latent = rng.normal(0.0, 1.0)
                    treat_prob = 1.0 / (
                        1.0
                        + np.exp(
                            -(
                                -0.2
                                + 0.35 * escs
                                + 0.45 * unmeasured_confounding * u_latent
                            )
                        )
                    )
                    treatment = int(rng.random() < treat_prob)
                    within_w = float(rng.uniform(0.85, 1.25))
                    final_w = school_weight * within_w
                    eps_student = rng.normal(0.0, sigma_e)

                    theta_true = (
                        true_intercept
                        + sys_intercept_shift
                        + true_escs * escs
                        + true_female * female
                        + beta_treat_sys * treatment
                        + true_school_escs * school_escs
                        + 35.0 * unmeasured_confounding * u_latent
                        + u_school
                        + eps_student
                    )
                    pvs = theta_true + rng.normal(0.0, pv_sd, size=n_pvs)

                    row: dict[str, Any] = {
                        "CNT": sys_code,
                        "STRATUM": stratum_id,
                        "PSU_IN_STRATUM": psu_idx,
                        "CNTSCHID": school_id,
                        "CNTSTUID": student_counter,
                        "W_FSTUWT": final_w,
                        "escs": float(escs),
                        "female": female,
                        "treatment": treatment,
                        "school_escs": float(school_escs),
                        "true_treatment_sys": float(beta_treat_sys),
                    }
                    for m_idx in range(n_pvs):
                        row[f"PV{m_idx + 1}MATH"] = float(pvs[m_idx])
                    records.append(row)
                    rep_weight_rows.append(final_w * rep_mult)

            global_stratum += 1

    df = pd.DataFrame.from_records(records)
    rep_mat = np.vstack(rep_weight_rows)
    rep_cols = {f"W_FSTURWT{r + 1}": rep_mat[:, r] for r in range(n_reps)}
    return pd.concat([df, pd.DataFrame(rep_cols, index=df.index)], axis=1)


def _naive_single_pv_ols(
    df: pd.DataFrame,
    predictors: list[str],
    outcome_col: str,
    weight_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute naive classical OLS (weighted or unweighted) SE ignoring clustering/PVs."""
    x_mat = np.column_stack([np.ones(len(df)), df[predictors].to_numpy(dtype=np.float64)])
    y_vec = df[outcome_col].to_numpy(dtype=np.float64)
    n_obs, p_dim = x_mat.shape
    if weight_col is None:
        w_vec = np.ones(n_obs, dtype=np.float64)
    else:
        w_vec = df[weight_col].to_numpy(dtype=np.float64)

    xtx_inv = np.linalg.inv(x_mat.T @ (w_vec[:, None] * x_mat))
    beta = xtx_inv @ (x_mat.T @ (w_vec * y_vec))
    resid = y_vec - x_mat @ beta
    rss = float(np.sum(w_vec * (resid ** 2)))
    w_sum = float(np.sum(w_vec))
    eff_scale = (rss / (w_sum * max(n_obs - p_dim, 1))) * n_obs
    se = np.sqrt(np.diag(xtx_inv) * eff_scale)
    if weight_col is None:
        sigma2 = rss / max(n_obs - p_dim, 1)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
    return beta, se


def run_coverage_benchmark(
    n_sim: int = 80,
    n_strata: int = 24,
    students_per_school: int = 16,
    n_pvs: int = 5,
    n_reps: int = 40,
    true_slope: float = 24.0,
    seed: int = 909,
) -> pd.DataFrame:
    """Evaluate empirical 95% confidence interval coverage across Monte Carlo trials.

    Compares ``pisakit`` (multi-PV Rubin + Fay BRR + Barnard-Rubin df) against
    two common erroneous baselines: unweighted classical OLS on PV1 and
    weighted classical OLS on PV1 (ignoring replicate clustering and PV variance).
    """
    from .models import fit_pv_ols
    from .types import DesignSpec

    design = DesignSpec.from_preset("pisa", n_reps=n_reps, n_pvs=n_pvs)
    predictors = ["escs", "female"]
    target_idx = 1  # slope on 'escs'

    methods = ["pisakit_pv_brr", "naive_weighted_pv1", "naive_unweighted_pv1"]
    stats: dict[str, dict[str, list[float]]] = {
        m: {"est": [], "se": [], "covered": [], "time": []} for m in methods
    }

    for sim_i in range(n_sim):
        df = simulate_lsa_survey(
            n_systems=1,
            n_strata_per_system=n_strata,
            students_per_school=students_per_school,
            n_pvs=n_pvs,
            n_reps=n_reps,
            fay_k=0.5,
            pv_sd=30.0,
            school_icc=0.35,
            true_escs=true_slope,
            true_school_escs=0.0,
            system_heterogeneity_sd=0.0,
            seed=seed + sim_i,
        )

        t0 = time.perf_counter()
        res = fit_pv_ols(
            data=df,
            outcome_pattern="PV{k}MATH",
            predictors=predictors,
            design=design,
        )
        dt = time.perf_counter() - t0
        row_escs = res.get_term("escs")
        stats["pisakit_pv_brr"]["est"].append(row_escs.estimate)
        stats["pisakit_pv_brr"]["se"].append(row_escs.se_total)
        stats["pisakit_pv_brr"]["covered"].append(
            float(row_escs.ci_low <= true_slope <= row_escs.ci_high)
        )
        stats["pisakit_pv_brr"]["time"].append(dt)

        for label, wcol in [("naive_weighted_pv1", "W_FSTUWT"), ("naive_unweighted_pv1", None)]:
            t1 = time.perf_counter()
            beta, se = _naive_single_pv_ols(df, predictors, "PV1MATH", weight_col=wcol)
            dt_n = time.perf_counter() - t1
            est_val = float(beta[target_idx])
            se_val = float(se[target_idx])
            ci_l = est_val - 1.96 * se_val
            ci_h = est_val + 1.96 * se_val
            stats[label]["est"].append(est_val)
            stats[label]["se"].append(se_val)
            stats[label]["covered"].append(float(ci_l <= true_slope <= ci_h))
            stats[label]["time"].append(dt_n)

    summary_rows: list[dict[str, Any]] = []
    for m in methods:
        est_arr = np.asarray(stats[m]["est"], dtype=np.float64)
        se_arr = np.asarray(stats[m]["se"], dtype=np.float64)
        cov_arr = np.asarray(stats[m]["covered"], dtype=np.float64)
        t_arr = np.asarray(stats[m]["time"], dtype=np.float64)
        emp_sd = float(np.std(est_arr, ddof=1)) if len(est_arr) > 1 else 0.0
        summary_rows.append(
            {
                "method": m,
                "n_sim": n_sim,
                "true_param": true_slope,
                "mean_estimate": float(np.mean(est_arr)),
                "bias": float(np.mean(est_arr) - true_slope),
                "empirical_sd": emp_sd,
                "mean_se": float(np.mean(se_arr)),
                "se_to_emp_ratio": float(np.mean(se_arr) / emp_sd) if emp_sd > 0 else np.nan,
                "coverage_95": float(np.mean(cov_arr)),
                "mean_runtime_ms": float(np.mean(t_arr) * 1000.0),
            }
        )
    return pd.DataFrame(summary_rows)
