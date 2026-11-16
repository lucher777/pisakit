"""
Core data structures, design specifications, and structured result containers
for plausible-value and replicate-weight survey inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DesignSpec:
    """Survey design specification describing weights, replicates, and PVs.

    Attributes:
        name: Human-readable design identifier (e.g., ``"pisa"``, ``"timss"``).
        final_weight: Column name for student final sampling weight.
        rep_prefix: Prefix for replicate weight columns.
        n_reps: Number of replicate weight columns.
        variance_method: Replicate variance estimator (``"fay_brr"``, ``"jk2"``, ``"jk1"``).
        fay_k: Fay's perturbation coefficient in ``[0, 1)`` used with ``"fay_brr"``.
        n_pvs: Number of plausible values per outcome construct.
        stratum_col: Optional explicit variance stratum column name.
        psu_col: Optional primary sampling unit (school) column name.
        index_start: Starting index for 1-based PV/replicate naming.
    """

    name: str = "pisa"
    final_weight: str = "W_FSTUWT"
    rep_prefix: str = "W_FSTURWT"
    n_reps: int = 80
    variance_method: str = "fay_brr"
    fay_k: float = 0.5
    n_pvs: int = 10
    stratum_col: str | None = "STRATUM"
    psu_col: str | None = "CNTSCHID"
    index_start: int = 1

    def __post_init__(self) -> None:
        if not self.final_weight:
            raise ValueError("final_weight must be a non-empty column name")
        if not self.rep_prefix:
            raise ValueError("rep_prefix must be a non-empty string")
        if self.n_reps < 2:
            raise ValueError(f"n_reps must be >= 2, got {self.n_reps}")
        if self.n_pvs < 1:
            raise ValueError(f"n_pvs must be >= 1, got {self.n_pvs}")
        if self.variance_method not in {"fay_brr", "jk2", "jk1"}:
            raise ValueError(
                f"Unsupported variance_method '{self.variance_method}'; "
                "expected one of 'fay_brr', 'jk2', 'jk1'"
            )
        if not (0.0 <= self.fay_k < 1.0):
            raise ValueError(f"fay_k must be in [0, 1), got {self.fay_k}")

    @classmethod
    def from_preset(cls, preset: str, **overrides: Any) -> "DesignSpec":
        """Create a DesignSpec from a named international assessment preset.

        Supported presets are ``"pisa"``, ``"timss"``, and ``"pirls"``.
        Keyword overrides replace any preset field values.
        """
        key = preset.strip().lower()
        presets: dict[str, dict[str, Any]] = {
            "pisa": {
                "name": "pisa",
                "final_weight": "W_FSTUWT",
                "rep_prefix": "W_FSTURWT",
                "n_reps": 80,
                "variance_method": "fay_brr",
                "fay_k": 0.5,
                "n_pvs": 10,
                "stratum_col": "STRATUM",
                "psu_col": "CNTSCHID",
                "index_start": 1,
            },
            "timss": {
                "name": "timss",
                "final_weight": "TOTWGT",
                "rep_prefix": "JKREP",
                "n_reps": 75,
                "variance_method": "jk2",
                "fay_k": 0.0,
                "n_pvs": 5,
                "stratum_col": "JKZONE",
                "psu_col": "IDSCHOOL",
                "index_start": 1,
            },
            "pirls": {
                "name": "pirls",
                "final_weight": "TOTWGT",
                "rep_prefix": "JKREP",
                "n_reps": 75,
                "variance_method": "jk2",
                "fay_k": 0.0,
                "n_pvs": 5,
                "stratum_col": "JKZONE",
                "psu_col": "IDSCHOOL",
                "index_start": 1,
            },
        }
        if key not in presets:
            valid = ", ".join(sorted(presets))
            raise ValueError(f"Unknown design preset '{preset}'. Valid presets: {valid}")
        params = dict(presets[key])
        params.update(overrides)
        return cls(**params)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the design specification to a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class VarianceDecomposition:
    """Imputation and sampling variance components for a scalar estimand."""

    q_bar: float
    u_bar: float
    b_imputation: float
    total_var: float
    se_total: float
    se_sampling: float
    se_imputation: float
    rvi: float
    fmi: float
    df_rubin: float
    df_complete: float
    df_br: float

    def to_dict(self) -> dict[str, float]:
        """Return variance components as a dictionary of floats."""
        return asdict(self)


@dataclass(frozen=True)
class CoefficientRow:
    """Single estimated regression or summary term after Rubin pooling."""

    term: str
    estimate: float
    se_total: float
    se_sampling: float
    se_imputation: float
    t_stat: float
    df: float
    p_value: float
    ci_low: float
    ci_high: float
    fmi: float
    rvi: float

    def to_dict(self) -> dict[str, Any]:
        """Convert this row to a dictionary."""
        return asdict(self)


@dataclass
class EstimationResult:
    """Structured result container returned by pisakit regression estimators."""

    model_type: str
    coefficients: list[CoefficientRow]
    n_obs: int
    n_clusters: int | None
    n_pvs: int
    n_reps: int
    variance_method: str
    fay_k: float
    alpha: float = 0.05
    r_squared: float | None = None
    fit_stats: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert coefficient rows into a tidy pandas DataFrame."""
        if not self.coefficients:
            return pd.DataFrame()
        return pd.DataFrame([row.to_dict() for row in self.coefficients])

    def get_term(self, term: str) -> CoefficientRow:
        """Look up a single term by exact name."""
        for row in self.coefficients:
            if row.term == term:
                return row
        available = [r.term for r in self.coefficients]
        raise KeyError(f"Term '{term}' not found; available terms: {available}")

    def format_summary(self, digits: int = 4) -> str:
        """Render a clean plain-text coefficient table with fit metadata."""
        lines = [
            f"Model: {self.model_type} | N={self.n_obs} | Clusters={self.n_clusters} | "
            f"PVs={self.n_pvs} | Reps={self.n_reps} ({self.variance_method}, k={self.fay_k})"
        ]
        if self.r_squared is not None:
            lines.append(f"Weighted R^2 (mean across PVs): {self.r_squared:.{digits}f}")
        for k, v in self.fit_stats.items():
            lines.append(f"  {k}: {v:.{digits}f}")
        header = (
            f"{'Term':<24} {'Est':>10} {'SE':>10} {'t':>8} "
            f"{'df':>9} {'p':>10} {'CI_low':>10} {'CI_high':>10} {'FMI':>7}"
        )
        lines.append("-" * len(header))
        lines.append(header)
        lines.append("-" * len(header))
        for r in self.coefficients:
            lines.append(
                f"{r.term:<24} {r.estimate:>10.{digits}f} {r.se_total:>10.{digits}f} "
                f"{r.t_stat:>8.3f} {r.df:>9.1f} {r.p_value:>10.4g} "
                f"{r.ci_low:>10.{digits}f} {r.ci_high:>10.{digits}f} {r.fmi:>7.3f}"
            )
        lines.append("-" * len(header))
        return "\n".join(lines)


@dataclass
class MetaPoolingResult:
    """Cross-system fixed-effect and DerSimonian-Laird random-effects summary."""

    term: str
    systems: list[str]
    pooled_fe: float
    se_fe: float
    ci_fe: tuple[float, float]
    p_fe: float
    pooled_re: float
    se_re: float
    ci_re: tuple[float, float]
    p_re: float
    tau2: float
    i2: float
    q_stat: float
    q_df: int
    q_pvalue: float
    system_table: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        """Serialize scalar pooling metrics (excluding DataFrame table)."""
        return {
            "term": self.term,
            "n_systems": len(self.systems),
            "pooled_fe": self.pooled_fe,
            "se_fe": self.se_fe,
            "ci_fe": list(self.ci_fe),
            "p_fe": self.p_fe,
            "pooled_re": self.pooled_re,
            "se_re": self.se_re,
            "ci_re": list(self.ci_re),
            "p_re": self.p_re,
            "tau2": self.tau2,
            "i2": self.i2,
            "q_stat": self.q_stat,
            "q_df": self.q_df,
            "q_pvalue": self.q_pvalue,
        }


@dataclass
class SensitivitySummary:
    """Cinelli-Hazlett OVB sensitivity diagnostics coupled to survey Rubin SE."""

    term: str
    estimate: float
    se: float
    df: float
    t_stat: float
    t_crit: float
    q_threshold: float
    alpha: float
    r2_yd_x: float
    rv_q: float
    rv_qa: float
    benchmark_bounds: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        """Serialize core OVB sensitivity statistics."""
        return {
            "term": self.term,
            "estimate": self.estimate,
            "se": self.se,
            "df": self.df,
            "t_stat": self.t_stat,
            "t_crit": self.t_crit,
            "q_threshold": self.q_threshold,
            "alpha": self.alpha,
            "r2_yd_x": self.r2_yd_x,
            "rv_q": self.rv_q,
            "rv_qa": self.rv_qa,
        }


@dataclass(frozen=True)
class EquivalenceReport:
    """Numerical equivalence and speed report against dense/SciPy reference solves.

    Attributes:
        wls_max_abs_err: Maximum absolute coefficient discrepancy between
            batched BLAS WLS and element-wise ``scipy.linalg.lstsq`` across all
            ``M * (1 + R)`` regressions on the bundled real PISA 2022 micro-slice.
        wls_batched_ms: Median runtime (ms) of ``_solve_batched_wls`` on the slice.
        wls_loop_ms: Median runtime (ms) of element-wise ``scipy.linalg.lstsq`` loops.
        wls_speedup: Ratio ``wls_loop_ms / wls_batched_ms``.
        brr_rubin_max_abs_err: Maximum discrepancy across pooled estimate, total SE,
            sampling SE, imputation SE, Barnard-Rubin df, and FMI relative to an
            explicit reference implementation of the OECD/intsvy Fay BRR + Rubin rules.
        reml_woodbury_max_abs_err: Maximum discrepancy between cluster-Woodbury REML
            and explicit dense block-diagonal covariance inversion for two-level REML.
        passed: True when all numerical tolerances are strictly satisfied.
    """

    wls_max_abs_err: float
    wls_batched_ms: float
    wls_loop_ms: float
    wls_speedup: float
    brr_rubin_max_abs_err: float
    reml_woodbury_max_abs_err: float
    passed: bool

    def summary(self) -> str:
        """Return a concise single-line numerical equivalence summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"EquivalenceReport[{status}]: "
            f"wls_err={self.wls_max_abs_err:.2e} ({self.wls_speedup:.1f}x speedup), "
            f"brr_rubin_err={self.brr_rubin_max_abs_err:.2e}, "
            f"reml_woodbury_err={self.reml_woodbury_max_abs_err:.2e}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize numerical equivalence metrics into a dictionary."""
        return asdict(self)
