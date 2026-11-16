"""
pisakit: Vectorized plausible-value and replicate-weight inference for
international large-scale educational assessments.
"""

from .datasets import (
    load_pisa2022_micro_slice,
    load_pisa2022_system_summary,
)
from .equiv import verify_numerical_equivalence
from .models import (
    fit_pv_multilevel,
    fit_pv_ols,
    fit_pv_quantile,
    fit_pv_within_between,
    pool_cross_system,
)
from .rubin import (
    combine_rubin,
    descriptive_pv_stats,
    joint_wald_test,
)
from .sensitivity import (
    adjusted_estimate,
    benchmark_confounder_bounds,
    compute_robustness_values,
    sensitivity_analysis,
    sensitivity_contour_grid,
)
from .synthetic import (
    build_hadamard_matrix,
    generate_fay_brr_matrix,
    run_coverage_benchmark,
    simulate_lsa_survey,
)
from .types import (
    CoefficientRow,
    DesignSpec,
    EquivalenceReport,
    EstimationResult,
    MetaPoolingResult,
    SensitivitySummary,
    VarianceDecomposition,
)
from .weights import (
    compute_senate_weights,
    effective_sample_size,
    get_design_preset,
    normalize_weights,
    prepare_domain_weights,
    replicate_variance,
    resolve_pv_columns,
    resolve_rep_columns,
    validate_survey_frame,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "CoefficientRow",
    "DesignSpec",
    "EquivalenceReport",
    "EstimationResult",
    "MetaPoolingResult",
    "SensitivitySummary",
    "VarianceDecomposition",
    "adjusted_estimate",
    "benchmark_confounder_bounds",
    "build_hadamard_matrix",
    "combine_rubin",
    "compute_robustness_values",
    "compute_senate_weights",
    "descriptive_pv_stats",
    "effective_sample_size",
    "fit_pv_multilevel",
    "fit_pv_ols",
    "fit_pv_quantile",
    "fit_pv_within_between",
    "generate_fay_brr_matrix",
    "get_design_preset",
    "joint_wald_test",
    "load_pisa2022_micro_slice",
    "load_pisa2022_system_summary",
    "normalize_weights",
    "pool_cross_system",
    "prepare_domain_weights",
    "replicate_variance",
    "resolve_pv_columns",
    "resolve_rep_columns",
    "run_coverage_benchmark",
    "sensitivity_analysis",
    "sensitivity_contour_grid",
    "simulate_lsa_survey",
    "validate_survey_frame",
    "verify_numerical_equivalence",
]
