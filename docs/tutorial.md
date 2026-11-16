# Tutorial: Real OECD PISA 2022 Analysis & Sensitivity Auditing

This tutorial walks through a complete reproducible workflow using the bundled OECD PISA 2022 micro-slice (`N = 960` students across `60` schools in `FIN`, `JPN`, `KOR`, and `EST`).

## 1. Load Bundled Real PISA 2022 Data

```python
from pisakit import DesignSpec, load_pisa2022_micro_slice, load_pisa2022_system_summary

df = load_pisa2022_micro_slice()
summary_15 = load_pisa2022_system_summary()
spec = DesignSpec.from_preset("pisa", n_pvs=5, n_reps=80)
print(df.shape)         # (960, 92)
print(summary_15.head())
```

## 2. Weighted PV OLS & Mundlak Within-Between Decomposition

```python
from pisakit import fit_pv_ols, fit_pv_within_between

ols = fit_pv_ols(
    data=df,
    outcome_pattern="PV{k}MATH",
    predictors=["escs", "female"],
    design=spec,
    system_col="CNT",
    add_system_fe=True,
    senate_weights=True,
)
print(ols.format_summary())

wb = fit_pv_within_between(
    data=df,
    outcome_pattern="PV{k}MATH",
    between_vars=["escs"],
    controls=["female"],
    design=spec,
    system_col="CNT",
    add_system_fe=True,
    senate_weights=True,
)
print(wb.get_term("escs_contextual"))
```

## 3. Two-Level REML, Cross-System Meta-Analysis, and OVB Sensitivity

```python
from pisakit import fit_pv_multilevel, pool_cross_system, sensitivity_analysis

mlm = fit_pv_multilevel(df[df["CNT"] == "FIN"], "PV{k}MATH", ["escs", "female"], design=spec)
print("FIN school ICC:", mlm.fit_stats["icc"])

meta = pool_cross_system(df, "PV{k}MATH", ["escs", "female"], term="escs", system_col="CNT", design=spec)
print("Pooled RE ESCS slope:", meta.pooled_re, "I^2:", meta.i2)

ovb = sensitivity_analysis(
    data=df,
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
```

## 4. Numerical Equivalence Verification & CLI

```bash
pisakit equiv
pisakit demo-real
pisakit benchmark --n-sim 80
```
