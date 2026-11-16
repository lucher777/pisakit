# pisakit

**Vectorized plausible-value, replicate-weight, multilevel, meta-analytic, and sensitivity inference for international large-scale educational assessments.**

`pisakit` provides a fast, strongly typed Python scientific pipeline for analyzing **PISA**, **TIMSS**, and **PIRLS** microdata, unifying $M$ plausible values (PVs) and $R$ replicate weights (Fay's BRR, JK2, JK1) under Rubin (1987) and Barnard-Rubin (1999) inference.

## Installation

```bash
pip install pisakit
```

For development and running tests:

```bash
pip install -e .[dev]
pytest -v
```

## Core Architecture

- **Design Presets & Weighting (`types.py`, `weights.py`)**: Presets for `pisa`, `timss`, and `pirls`; Senate weight rescaling (`compute_senate_weights`); Asparouhov level-1 cluster weight scaling (`normalize_weights`); Kish effective sample sizes; and unbiased zero-weight domain masks (`prepare_domain_weights`).
- **Bundled Real Benchmark Data (`datasets.py`)**: Includes a de-identified 960-student OECD PISA 2022 micro-slice (`FIN`, `JPN`, `KOR`, `EST`) with 5 mathematics PVs and 80 Fay BRR weights, plus a 15-system full-sample benchmark summary.
- **Batched BLAS & Woodbury Estimators (`models.py`)**: Rank-2 BLAS contractions across all $M \times (1+R)$ pairs (`fit_pv_ols`, `fit_pv_within_between`, `fit_pv_quantile`), plus rank-1 cluster-Woodbury two-level random-intercept REML (`fit_pv_multilevel`) and DerSimonian-Laird cross-system pooling (`pool_cross_system`).
- **Rubin & Sensitivity Engine (`rubin.py`, `sensitivity.py`)**: Scalar and multivariate $D_1$ Wald inference (`joint_wald_test`) coupled directly to Cinelli-Hazlett OVB robustness bounds (`RV_q`, `RV_{q,alpha}`, covariate benchmarking, and 2D contour grids).
- **Numerical Parity Verification (`equiv.py`)**: Automated machine-precision cross-checks against `scipy.linalg.lstsq`, from-scratch OECD Fay BRR + Barnard-Rubin formulas, and explicit dense block-diagonal REML covariance inversion.
