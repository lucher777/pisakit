"""
Survey design presets, column resolution, replicate variance estimation,
Senate weight rescaling, hierarchical weight normalization, and domain masks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import DesignSpec


def get_design_preset(preset: str, **overrides: object) -> DesignSpec:
    """Return a validated DesignSpec from a built-in assessment preset."""
    return DesignSpec.from_preset(preset, **overrides)


def resolve_pv_columns(
    data: pd.DataFrame,
    outcome_pattern: str,
    n_pvs: int | None = None,
    index_start: int = 1,
) -> list[str]:
    """Resolve plausible-value column names from a pattern and DataFrame.

    Supports placeholder syntax (e.g., ``"PV{k}MATH"``) as well as suffix
    concatenation (e.g., ``"ASMMA"`` resolving ``"ASMMA01"`` / ``"ASMMA1"``
    or prefix+integer when ``"{k}"`` is omitted).
    """
    cols = set(data.columns)
    resolved: list[str] = []

    if "{k}" in outcome_pattern:
        if n_pvs is not None:
            for k in range(index_start, index_start + n_pvs):
                col = outcome_pattern.format(k=k)
                if col not in cols:
                    raise KeyError(
                        f"Plausible value column '{col}' resolved from '{outcome_pattern}' "
                        "not found in DataFrame."
                    )
                resolved.append(col)
        else:
            k = index_start
            while outcome_pattern.format(k=k) in cols:
                resolved.append(outcome_pattern.format(k=k))
                k += 1
    else:
        if outcome_pattern in cols and n_pvs == 1:
            return [outcome_pattern]
        k = index_start
        limit = (index_start + n_pvs) if n_pvs is not None else (index_start + 100)
        while k < limit:
            cand_plain = f"{outcome_pattern}{k}"
            cand_pad = f"{outcome_pattern}{k:02d}"
            if cand_plain in cols:
                resolved.append(cand_plain)
            elif cand_pad in cols:
                resolved.append(cand_pad)
            elif n_pvs is not None:
                raise KeyError(
                    f"Missing PV column '{cand_plain}' (or '{cand_pad}') in DataFrame."
                )
            else:
                break
            k += 1

    if not resolved:
        raise ValueError(
            f"Could not find any plausible value columns matching pattern '{outcome_pattern}'."
        )
    return resolved


def resolve_rep_columns(data: pd.DataFrame, design: DesignSpec) -> list[str]:
    """Resolve replicate weight columns defined by ``design``."""
    cols = set(data.columns)
    rep_cols: list[str] = []
    for idx in range(design.index_start, design.index_start + design.n_reps):
        col = f"{design.rep_prefix}{idx}"
        if col not in cols:
            raise KeyError(
                f"Expected replicate weight column '{col}' (preset={design.name}) "
                "is missing from DataFrame."
            )
        rep_cols.append(col)
    return rep_cols


def validate_survey_frame(
    data: pd.DataFrame,
    design: DesignSpec,
    outcome_pattern: str | None = None,
    required_cols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Validate weight columns, replicate columns, PVs, and model variables.

    Returns ``(pv_cols, rep_cols)`` after confirming finite non-negative weights
    with strictly positive total weight.
    """
    if data.empty:
        raise ValueError("Input survey DataFrame is empty.")
    if design.final_weight not in data.columns:
        raise KeyError(f"Final weight column '{design.final_weight}' not in DataFrame.")

    w_full = data[design.final_weight].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(w_full)):
        raise ValueError(f"Final weight column '{design.final_weight}' contains NaN or Inf.")
    if np.any(w_full < 0.0):
        raise ValueError(f"Final weight column '{design.final_weight}' contains negative weights.")
    if float(np.sum(w_full)) <= 0.0:
        raise ValueError(f"Total sum of '{design.final_weight}' must be strictly positive.")

    rep_cols = resolve_rep_columns(data, design)
    w_reps = data[rep_cols].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(w_reps)):
        raise ValueError("Replicate weight columns contain NaN or Inf.")
    if np.any(w_reps < 0.0):
        raise ValueError("Replicate weight columns contain negative weights.")

    if required_cols:
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise KeyError(f"Missing required analysis columns: {missing}")

    pv_cols: list[str] = []
    if outcome_pattern is not None:
        pv_cols = resolve_pv_columns(
            data,
            outcome_pattern,
            n_pvs=design.n_pvs,
            index_start=design.index_start,
        )
        y_mat = data[pv_cols].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(y_mat)):
            raise ValueError("Outcome plausible-value columns contain NaN or Inf.")

    return pv_cols, rep_cols


def _variance_multiplier(n_reps: int, method: str, fay_k: float) -> float:
    """Return the scalar multiplier for replicate variance estimation."""
    if method == "fay_brr":
        return 1.0 / (n_reps * ((1.0 - fay_k) ** 2))
    if method == "jk2":
        return 1.0
    if method == "jk1":
        return (n_reps - 1.0) / n_reps
    raise ValueError(f"Unsupported replicate variance method: '{method}'")


def replicate_variance(
    theta_full: np.ndarray | float,
    theta_reps: np.ndarray,
    method: str = "fay_brr",
    fay_k: float = 0.5,
) -> np.ndarray:
    """Compute survey sampling variance from replicate parameter estimates.

    Args:
        theta_full: Full-sample estimate (scalar or array of arbitrary shape).
        theta_reps: Replicate estimates with replicate axis at position 0, i.e.,
            shape ``(n_reps, ...)``.
        method: One of ``"fay_brr"``, ``"jk2"``, or ``"jk1"``.
        fay_k: Fay perturbation factor in ``[0, 1)`` for ``"fay_brr"``.

    Returns:
        Array of sampling variances matching the shape of ``theta_full``.
    """
    reps = np.asarray(theta_reps, dtype=np.float64)
    full = np.asarray(theta_full, dtype=np.float64)
    if reps.ndim < 1:
        raise ValueError("theta_reps must have at least 1 dimension (n_reps, ...).")
    n_reps = reps.shape[0]
    if n_reps < 2:
        raise ValueError(f"At least 2 replicates required, got {n_reps}.")
    diff = reps - full[None, ...]
    factor = _variance_multiplier(n_reps, method, fay_k)
    return factor * np.sum(diff * diff, axis=0)


def replicate_covariance(
    theta_full: np.ndarray,
    theta_reps: np.ndarray,
    method: str = "fay_brr",
    fay_k: float = 0.5,
) -> np.ndarray:
    """Compute the full ``(p, p)`` replicate sampling covariance matrix."""
    full = np.asarray(theta_full, dtype=np.float64).reshape(-1)
    reps = np.asarray(theta_reps, dtype=np.float64)
    if reps.ndim != 2 or reps.shape[1] != full.shape[0]:
        raise ValueError(
            f"Shape mismatch: theta_full has shape {full.shape}, theta_reps has {reps.shape}."
        )
    diff = reps - full[None, :]
    factor = _variance_multiplier(reps.shape[0], method, fay_k)
    return factor * (diff.T @ diff)


def compute_senate_weights(
    data: pd.DataFrame,
    design: DesignSpec,
    system_col: str,
    target_sum: float = 500.0,
    inplace: bool = False,
) -> pd.DataFrame:
    """Rescale final and replicate weights so every system sums to ``target_sum``.

    Scaling both ``design.final_weight`` and all ``design.rep_prefix*`` columns by
    the identical per-system constant preserves within-system Fay BRR / JK2
    relative replicate ratios while eliminating population-size dominance in
    pooled cross-national regressions.
    """
    if target_sum <= 0.0:
        raise ValueError(f"target_sum must be positive, got {target_sum}")
    if system_col not in data.columns:
        raise KeyError(f"System column '{system_col}' not found in DataFrame.")

    _, rep_cols = validate_survey_frame(data, design)
    out = data if inplace else data.copy()

    w_sums = out.groupby(system_col, sort=False)[design.final_weight].transform("sum")
    if np.any(w_sums.to_numpy(dtype=np.float64) <= 0.0):
        raise ValueError("Every system must have a strictly positive sum of final weights.")

    scale = target_sum / w_sums.to_numpy(dtype=np.float64)
    weight_cols = [design.final_weight] + rep_cols
    out[weight_cols] = out[weight_cols].to_numpy(dtype=np.float64) * scale[:, None]
    return out


def normalize_weights(
    data: pd.DataFrame,
    weight_col: str,
    group_col: str | None = None,
    method: str = "mean_one",
) -> np.ndarray:
    """Normalize student sampling weights globally or within level-2 clusters.

    Supported methods:
        - ``"mean_one"``: scales weights within each group so their sum equals
          the cluster sample size ``n_j`` (Asparouhov 2006 / PISA Method A).
        - ``"sum_one"``: scales weights within each group to sum to ``1.0``.
        - ``"effective"``: scales weights within each group so their sum equals
          the Kish effective sample size ``(sum w)^2 / sum(w^2)``.
    """
    if weight_col not in data.columns:
        raise KeyError(f"Weight column '{weight_col}' not found.")
    w = data[weight_col].to_numpy(dtype=np.float64)
    if np.any(w < 0.0):
        raise ValueError("Weights cannot be negative for normalization.")

    valid_methods = {"mean_one", "sum_one", "effective"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}, got '{method}'")

    if group_col is None:
        groups = np.zeros(len(w), dtype=np.int64)
        inv = groups
        n_groups = 1
    else:
        if group_col not in data.columns:
            raise KeyError(f"Group column '{group_col}' not found.")
        _, inv = np.unique(data[group_col].to_numpy(), return_inverse=True)
        n_groups = int(inv.max()) + 1

    sum_w = np.bincount(inv, weights=w, minlength=n_groups)
    if np.any(sum_w <= 0.0):
        raise ValueError("All clusters/groups must have positive weight sums.")

    if method == "sum_one":
        denom = sum_w[inv]
        return w / denom
    if method == "mean_one":
        counts = np.bincount(inv, minlength=n_groups).astype(np.float64)
        scale = (counts / sum_w)[inv]
        return w * scale

    sum_w2 = np.bincount(inv, weights=w * w, minlength=n_groups)
    n_eff = (sum_w * sum_w) / np.maximum(sum_w2, 1e-18)
    scale = (n_eff / sum_w)[inv]
    return w * scale


def effective_sample_size(weights: np.ndarray) -> tuple[float, float]:
    """Compute Kish effective sample size and design effect on positive weights.

    Returns ``(n_eff, deff)`` where ``deff = n_positive / n_eff``.
    """
    w = np.asarray(weights, dtype=np.float64)
    pos = w[w > 0.0]
    if pos.size == 0:
        raise ValueError("No strictly positive weights found.")
    sum_w = float(np.sum(pos))
    sum_w2 = float(np.sum(pos * pos))
    n_eff = (sum_w * sum_w) / sum_w2
    deff = float(pos.size) / n_eff
    return n_eff, deff


def prepare_domain_weights(
    data: pd.DataFrame,
    mask: np.ndarray | pd.Series,
    design: DesignSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out final and replicate weights outside a subpopulation domain.

    Keeping all rows with zero weights outside ``mask`` preserves complex
    variance strata and ensures unbiased Fay BRR / JK2 domain standard errors.
    """
    _, rep_cols = validate_survey_frame(data, design)
    bool_mask = np.asarray(mask, dtype=bool)
    if bool_mask.shape != (len(data),):
        raise ValueError(
            f"Domain mask length {bool_mask.shape} does not match DataFrame length {len(data)}."
        )
    if not np.any(bool_mask):
        raise ValueError("Domain mask selects zero observations.")

    indicator = bool_mask.astype(np.float64)
    w_full = data[design.final_weight].to_numpy(dtype=np.float64) * indicator
    if float(np.sum(w_full)) <= 0.0:
        raise ValueError("Selected domain has zero positive sampling weight.")
    w_reps = data[rep_cols].to_numpy(dtype=np.float64) * indicator[:, None]
    return w_full, w_reps
