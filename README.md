# pisakit

**Vectorized plausible-value, replicate-weight, multilevel, meta-analytic, and sensitivity inference for international large-scale educational assessments.**

`pisakit` provides a clean, fast, pure-Python/NumPy/SciPy scientific workflow for analyzing complex survey assessments such as **PISA**, **TIMSS**, and **PIRLS**, where student proficiency is represented by $M$ plausible values (PVs) and complex multistage sampling variance is captured via $R$ replicate weights (Fay's BRR or paired jackknife JK2/JK1).

## Key Features

- **Design-aware survey weighting**: Built-in design presets (`pisa`, `timss`, `pirls`), automatic detection of PV and replicate columns, Senate weight rescaling across countries, Asparouhov-style hierarchical weight normalization, Kish effective sample size ($N_{\text{eff}}$), and proper zero-weighted subpopulation domain estimation.
- **Exact Rubin & Barnard-Rubin pooling**: Full scalar and multivariate combining rules across $M$ PVs with imputation variance ($B$), sampling variance ($\bar{U}$), fraction of missing information (FMI), relative variance increase (RVI), Barnard-Rubin finite-sample degrees of freedom ($\nu^*$), and multi-parameter $D_1$ Wald tests.
- **Vectorized BLAS contraction for PV regressions**: Solves all $M \times (1 + R)$ weighted regressions via rank-2 BLAS matrix contractions in `fit_pv_ols`, `fit_pv_within_between` (cluster mean decomposition and contextual tests), and `fit_pv_quantile` (IRLS quantile regression across $\tau \in (0, 1)$).
- **Weighted two-level random-intercept modeling**: `fit_pv_multilevel` computes two-level GLS/REML variance components ($\tau_0^2$, $\sigma^2$, ICC) using rank-1 cluster Woodbury updates pooled across all PVs and replicate weights.
- **Cross-system meta-analysis**: `pool_cross_system` estimates system-specific effects and aggregates them using inverse-variance fixed-effect and DerSimonian-Laird random-effects models ($\tau^2$, $I^2$, Cochran's $Q$).
- **Survey-coupled Cinelli-Hazlett OVB sensitivity**: `sensitivity_analysis` computes partial-$R^2$ bounds, $RV_q$, $RV_{q,\alpha}$, confounder-adjusted CIs, covariate benchmarking, and 2D contour grids directly from Rubin-BRR degrees of freedom.
- **Bundled real OECD PISA 2022 data & equivalence suite**: Ships with a de-identified 960-student PISA 2022 micro-slice (`FIN`, `JPN`, `KOR`, `EST`), a 15-system full-sample PISA 2022 summary table, and automated numerical parity checks (`verify_numerical_equivalence`) against SciPy WLS, OECD Fay BRR formulas, and explicit dense block-diagonal REML.

## Installation

```bash
pip install -e .
```

For development, testing, and documentation tools:

```bash
pip install -e .[dev]
pytest -v
```

## Quickstart (Real OECD PISA 2022 Slice)

```python
from pisakit import (
    DesignSpec,
    load_pisa2022_micro_slice,
    fit_pv_ols,
    pool_cross_system,
    sensitivity_analysis,
    verify_numerical_equivalence,
)

data = load_pisa2022_micro_slice()
spec = DesignSpec.from_preset("pisa", n_reps=80, n_pvs=5)

res = fit_pv_ols(
    data=data,
    outcome_pattern="PV{k}MATH",
    predictors=["escs", "female"],
    design=spec,
    system_col="CNT",
    add_system_fe=True,
    senate_weights=True,
)
print(res.format_summary())

meta = pool_cross_system(
    data=data,
    outcome_pattern="PV{k}MATH",
    predictors=["escs", "female"],
    term="escs",
    system_col="CNT",
    design=spec,
)
print(f"Pooled RE ESCS slope: {meta.pooled_re:.3f} (SE={meta.se_re:.3f}, I2={meta.i2:.1%})")

ovb = sensitivity_analysis(
    data=data,
    outcome_pattern="PV{k}MATH",
    treatment="escs",
    controls=["female"],
    design=spec,
    benchmarks=["female"],
    system_col="CNT",
    add_system_fe=True,
    senate_weights=True,
)
print(f"RV_q={ovb.rv_q:.4f}, RV_qa={ovb.rv_qa:.4f}")
print(verify_numerical_equivalence(data).summary())
```

## Command-Line Interface

```bash
pisakit info
pisakit demo-real
pisakit equiv
pisakit benchmark --n-sim 80
pisakit simulate --out sim_lsa.csv --n-systems 3
```

## Documentation & Contributing

Full tutorials and API documentation are maintained in `docs/` (`mkdocs.yml`). See `CONTRIBUTING.md` for contributor guidelines.

## License

MIT License. See `LICENSE` for details.
