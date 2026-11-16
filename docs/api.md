# API Reference

## Design & Data Structures (`pisakit.types`)

- `DesignSpec(name, final_weight, rep_prefix, n_reps, variance_method, fay_k, n_pvs, stratum_col, psu_col, index_start)`: Survey specification dataclass with `.from_preset("pisa" | "timss" | "pirls")` and `.to_dict()`.
- `EstimationResult`: Structured output returned by regression and descriptive estimators, providing `.to_dataframe()`, `.get_term(term)`, and `.format_summary()`.
- `CoefficientRow`: Per-term Rubin + Barnard-Rubin statistics (`estimate`, `se_total`, `se_sampling`, `se_imputation`, `t_stat`, `df`, `p_value`, `ci_low`, `ci_high`, `fmi`, `rvi`).
- `VarianceDecomposition`: Detailed variance components (`q_bar`, `u_bar`, `b_imputation`, `total_var`, `df_rubin`, `df_complete`, `df_br`).
- `MetaPoolingResult`: Fixed- and random-effects meta-pooling container (`pooled_fe`, `pooled_re`, `tau2`, `i2`, `q_stat`, `system_table`).
- `SensitivitySummary`: Survey-coupled OVB robustness summary (`rv_q`, `rv_qa`, `r2_yd_x`, `t_crit`, `benchmark_bounds`).
- `EquivalenceReport`: Cross-package numerical parity report (`wls_max_abs_err`, `wls_speedup`, `brr_rubin_max_abs_err`, `reml_woodbury_max_abs_err`, `passed`).

## Survey Weights & Validation (`pisakit.weights`)

- `get_design_preset(preset, **overrides)`
- `resolve_pv_columns(data, outcome_pattern, n_pvs=None, index_start=1)`
- `resolve_rep_columns(data, design)`
- `validate_survey_frame(data, design, outcome_pattern=None, required_cols=None)`
- `replicate_variance(theta_full, theta_reps, method="fay_brr", fay_k=0.5)`
- `compute_senate_weights(data, design, system_col, target_sum=500.0, inplace=False)`
- `normalize_weights(data, weight_col, group_col=None, method="mean_one")`
- `effective_sample_size(weights)`
- `prepare_domain_weights(data, mask, design)`

## Rubin Rules & Descriptives (`pisakit.rubin`)

- `combine_rubin(q_m, u_m, df_complete=None, alpha=0.05)`
- `joint_wald_test(q_m, cov_m, term_indices=None)`
- `descriptive_pv_stats(data, outcome_pattern, design, by=None, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), reference_group=None, alpha=0.05)`

## Estimators (`pisakit.models`)

- `fit_pv_ols(data, outcome_pattern, predictors, design, system_col=None, add_system_fe=False, senate_weights=False, senate_target=500.0, domain_mask=None, alpha=0.05)`
- `fit_pv_within_between(data, outcome_pattern, between_vars, cluster_col=None, controls=(), design=None, system_col=None, add_system_fe=False, senate_weights=False, alpha=0.05)`
- `fit_pv_quantile(data, outcome_pattern, predictors, design, tau=0.5, system_col=None, add_system_fe=False, senate_weights=False, max_iter=35, alpha=0.05)`
- `fit_pv_multilevel(data, outcome_pattern, predictors, cluster_col=None, design=None, weight_scaling="mean_one", system_col=None, add_system_fe=False, alpha=0.05)`
- `pool_cross_system(data, outcome_pattern, predictors, term, system_col="CNT", design=None, estimator=None, estimator_kwargs=None, alpha=0.05)`

## Sensitivity & Equivalence (`pisakit.sensitivity`, `pisakit.equiv`)

- `compute_robustness_values(t_stat, df, q=1.0, alpha=0.05)`
- `adjusted_estimate(estimate, se, df, r2_dz_x, r2_yz_dx, direction=-1.0, alpha=0.05)`
- `benchmark_confounder_bounds(data, outcome_pattern, treatment, controls, benchmarks, design, base_result=None, multipliers=(1.0, 2.0, 3.0), ...)`
- `sensitivity_contour_grid(estimate, se, df, max_r2_dz=0.25, max_r2_yz=0.25, n_grid=60, alpha=0.05)`
- `sensitivity_analysis(data, outcome_pattern, treatment, controls, design=None, benchmarks=(), ...)`
- `verify_numerical_equivalence(data=None, wls_tol=1e-10, rubin_tol=1e-11, reml_tol=1e-10)`

## Bundled Data & Simulation (`pisakit.datasets`, `pisakit.synthetic`)

- `load_pisa2022_micro_slice()`
- `load_pisa2022_system_summary()`
- `build_hadamard_matrix(order)`
- `generate_fay_brr_matrix(n_strata, fay_k=0.5)`
- `simulate_lsa_survey(...)`
- `run_coverage_benchmark(...)`
