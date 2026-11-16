"""
End-to-end quickstart demonstrating pisakit on bundled real OECD PISA 2022 data
and cross-package numerical equivalence checks: weighted PV OLS, Mundlak
within-between decomposition, two-level Woodbury REML, cross-system meta-analysis,
and survey-coupled Cinelli-Hazlett sensitivity.
"""

from __future__ import annotations

from pisakit import (
    DesignSpec,
    fit_pv_multilevel,
    fit_pv_ols,
    fit_pv_within_between,
    load_pisa2022_micro_slice,
    load_pisa2022_system_summary,
    pool_cross_system,
    sensitivity_analysis,
    verify_numerical_equivalence,
)


def main() -> None:
    """Run all core pisakit estimators on real PISA 2022 data and verify parity."""
    print("1. Bundled OECD PISA 2022 Micro-Slice (FIN, JPN, KOR, EST; N=960)")
    df = load_pisa2022_micro_slice()
    spec = DesignSpec.from_preset("pisa", n_reps=80, n_pvs=5)

    ols_res = fit_pv_ols(
        data=df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female"],
        design=spec,
        system_col="CNT",
        add_system_fe=True,
        senate_weights=True,
    )
    print(ols_res.format_summary())

    wb_res = fit_pv_within_between(
        data=df,
        outcome_pattern="PV{k}MATH",
        between_vars=["escs"],
        controls=["female"],
        design=spec,
        system_col="CNT",
        add_system_fe=True,
        senate_weights=True,
    )
    ctx_row = wb_res.get_term("escs_contextual")
    print(
        f"\nESCS contextual effect: {ctx_row.estimate:.3f} "
        f"(SE={ctx_row.se_total:.3f}, p={ctx_row.p_value:.4g})"
    )

    mlm_res = fit_pv_multilevel(
        data=df[df["CNT"] == "FIN"],
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female"],
        design=spec,
    )
    print(f"FIN two-level REML school ICC: {mlm_res.fit_stats['icc']:.4f}")

    meta_res = pool_cross_system(
        data=df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female"],
        term="escs",
        system_col="CNT",
        design=spec,
    )
    print(
        f"Cross-system RE pooled ESCS slope: {meta_res.pooled_re:.3f} "
        f"(SE={meta_res.se_re:.3f}, I2={meta_res.i2:.1%})"
    )

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
    print(f"OVB Robustness Values: RV_q={ovb.rv_q:.4f}, RV_qa={ovb.rv_qa:.4f}")

    summary_15 = load_pisa2022_system_summary()
    print(f"\n2. Loaded 15-system full-sample PISA 2022 summary ({len(summary_15)} countries)")

    print("\n3. Cross-Package Numerical Equivalence & BLAS Speedup")
    report = verify_numerical_equivalence(df)
    print(f"   {report.summary()}")


if __name__ == "__main__":
    main()
