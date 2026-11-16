"""
Command-line entry point for pisakit: survey simulation, PV regressions,
cross-system meta-pooling, OVB sensitivity checks, and Monte Carlo benchmarking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from . import __version__
from .datasets import load_pisa2022_micro_slice, load_pisa2022_system_summary
from .equiv import verify_numerical_equivalence
from .models import fit_pv_multilevel, fit_pv_ols, fit_pv_within_between, pool_cross_system
from .sensitivity import sensitivity_analysis
from .synthetic import run_coverage_benchmark, simulate_lsa_survey
from .types import DesignSpec


def _parse_csv_list(raw: str) -> list[str]:
    """Split a comma-separated CLI argument into non-empty strings."""
    return [item.strip() for item in raw.split(",") if item.strip()]


def _cmd_info(args: argparse.Namespace) -> int:
    """Print package version and built-in assessment presets."""
    presets = {name: DesignSpec.from_preset(name).to_dict() for name in ("pisa", "timss", "pirls")}
    payload = {"package": "pisakit", "version": __version__, "presets": presets}
    if args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        print(f"pisakit {__version__}")
        for name, spec in presets.items():
            print(
                f"  - {name}: w={spec['final_weight']}, rep={spec['rep_prefix']}[1..{spec['n_reps']}], "
                f"method={spec['variance_method']}, k={spec['fay_k']}, n_pvs={spec['n_pvs']}"
            )
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    """Generate a synthetic LSA dataset and save to CSV."""
    df = simulate_lsa_survey(
        n_systems=args.n_systems,
        n_strata_per_system=args.n_strata,
        students_per_school=args.students_per_school,
        n_pvs=args.n_pvs,
        n_reps=args.n_reps,
        seed=args.seed,
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    msg = {
        "rows": len(df),
        "cols": df.shape[1],
        "systems": int(df["CNT"].nunique()),
        "out": str(out_path),
    }
    print(json.dumps(msg, indent=2) if args.format == "json" else f"Wrote {len(df)} rows to {out_path}")
    return 0


def _cmd_ols(args: argparse.Namespace) -> int:
    """Run weighted PV OLS regression from a CSV survey dataset."""
    df = pd.read_csv(args.data)
    spec = DesignSpec.from_preset(args.design, n_reps=args.n_reps, n_pvs=args.n_pvs)
    res = fit_pv_ols(
        data=df,
        outcome_pattern=args.outcome,
        predictors=_parse_csv_list(args.predictors),
        design=spec,
        system_col=args.system_col,
        add_system_fe=args.system_fe,
        senate_weights=args.senate,
    )
    if args.format == "json":
        print(json.dumps({"r_squared": res.r_squared, "coefficients": res.to_dataframe().to_dict("records")}, indent=2))
    else:
        print(res.format_summary())
    return 0


def _cmd_meta(args: argparse.Namespace) -> int:
    """Run cross-system fixed- and random-effects pooling for a single focal term."""
    df = pd.read_csv(args.data)
    spec = DesignSpec.from_preset(args.design, n_reps=args.n_reps, n_pvs=args.n_pvs)
    meta = pool_cross_system(
        data=df,
        outcome_pattern=args.outcome,
        predictors=_parse_csv_list(args.predictors),
        term=args.term,
        system_col=args.system_col,
        design=spec,
    )
    if args.format == "json":
        print(json.dumps(meta.to_dict(), indent=2))
    else:
        print(
            f"Meta Pooling [{meta.term}] across {len(meta.systems)} systems: "
            f"RE={meta.pooled_re:.4f} (SE={meta.se_re:.4f}, p={meta.p_re:.4g}), "
            f"FE={meta.pooled_fe:.4f}, tau2={meta.tau2:.4f}, I2={meta.i2:.1%}"
        )
    return 0


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    """Run survey-coupled Cinelli-Hazlett OVB sensitivity analysis."""
    df = pd.read_csv(args.data)
    spec = DesignSpec.from_preset(args.design, n_reps=args.n_reps, n_pvs=args.n_pvs)
    ovb = sensitivity_analysis(
        data=df,
        outcome_pattern=args.outcome,
        treatment=args.treatment,
        controls=_parse_csv_list(args.controls),
        design=spec,
        benchmarks=_parse_csv_list(args.benchmarks),
        system_col=args.system_col,
        add_system_fe=args.system_fe,
    )
    if args.format == "json":
        payload = ovb.to_dict()
        payload["benchmarks"] = ovb.benchmark_bounds.to_dict("records")
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"OVB Sensitivity [{ovb.term}]: est={ovb.estimate:.4f}, se={ovb.se:.4f}, "
            f"t={ovb.t_stat:.3f}, df={ovb.df:.1f}, RV_q={ovb.rv_q:.4f}, RV_qa={ovb.rv_qa:.4f}"
        )
        if not ovb.benchmark_bounds.empty:
            print(ovb.benchmark_bounds.to_string(index=False))
    return 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    """Execute Monte Carlo 95% coverage benchmark against naive baselines."""
    table = run_coverage_benchmark(n_sim=args.n_sim, seed=args.seed)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_path, index=False)
    if args.format == "json":
        print(json.dumps(table.to_dict("records"), indent=2))
    else:
        print(table.to_string(index=False))
    return 0


def _cmd_equiv(args: argparse.Namespace) -> int:
    """Run machine-precision numerical equivalence checks on real PISA 2022 data."""
    report = verify_numerical_equivalence()
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
        print(
            f"Batched BLAS timing (405 regressions, N=960): "
            f"{report.wls_batched_ms:.2f} ms vs SciPy lstsq loop {report.wls_loop_ms:.2f} ms "
            f"({report.wls_speedup:.1f}x faster)"
        )
    return 0 if report.passed else 1


def _cmd_demo_real(args: argparse.Namespace) -> int:
    """Demonstrate pisakit on the bundled OECD PISA 2022 micro-slice and 15-system table."""
    df = load_pisa2022_micro_slice()
    spec = DesignSpec.from_preset("pisa", n_pvs=5, n_reps=80)
    ols = fit_pv_ols(
        df,
        outcome_pattern="PV{k}MATH",
        predictors=["escs", "female"],
        design=spec,
        system_col="CNT",
        add_system_fe=True,
        senate_weights=True,
    )
    wb = fit_pv_within_between(
        df,
        outcome_pattern="PV{k}MATH",
        between_vars=["escs"],
        controls=["female"],
        design=spec,
        system_col="CNT",
        add_system_fe=True,
        senate_weights=True,
    )
    ovb = sensitivity_analysis(
        df,
        outcome_pattern="PV{k}MATH",
        treatment="escs",
        controls=["female"],
        design=spec,
        benchmarks=["female"],
        system_col="CNT",
        add_system_fe=True,
        senate_weights=True,
    )
    summary_15 = load_pisa2022_system_summary()
    if args.format == "json":
        payload = {
            "n_slice": len(df),
            "ols_escs": ols.get_term("escs").to_dict(),
            "wb_contextual": wb.get_term("escs_contextual").to_dict(),
            "ovb": ovb.to_dict(),
            "systems_15": summary_15.to_dict("records"),
        }
        print(json.dumps(payload, indent=2))
    else:
        print("=== Bundled OECD PISA 2022 Micro-Slice (FIN, JPN, KOR, EST; N=960) ===")
        print(ols.format_summary())
        ctx = wb.get_term("escs_contextual")
        print(
            f"Mundlak contextual ESCS effect: {ctx.estimate:.3f} "
            f"(SE={ctx.se_total:.3f}, p={ctx.p_value:.4g})"
        )
        print(
            f"OVB Robustness (treatment=escs): RV_q={ovb.rv_q:.4f}, RV_qa={ovb.rv_qa:.4f}"
        )
        print("\n=== Full-Sample PISA 2022 15-System Summary (M=10 PVs, R=80 BRR) ===")
        print(
            summary_15[["cnt", "n_students", "mean_math", "beta_escs", "se_escs", "beta_within", "beta_between", "rv_q"]]
            .head(8)
            .to_string(index=False)
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for pisakit."""
    parser = argparse.ArgumentParser(
        prog="pisakit",
        description="Plausible-value and replicate-weight inference for large-scale assessments.",
    )
    parser.add_argument("--version", action="version", version=f"pisakit {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser("info", help="Show version and assessment design presets.")
    p_info.add_argument("--format", choices=["text", "json"], default="text")
    p_info.set_defaults(func=_cmd_info)

    p_sim = sub.add_parser("simulate", help="Generate a synthetic LSA dataset.")
    p_sim.add_argument("--out", required=True, help="Output CSV path.")
    p_sim.add_argument("--n-systems", type=int, default=3)
    p_sim.add_argument("--n-strata", type=int, default=12)
    p_sim.add_argument("--students-per-school", type=int, default=15)
    p_sim.add_argument("--n-pvs", type=int, default=5)
    p_sim.add_argument("--n-reps", type=int, default=40)
    p_sim.add_argument("--seed", type=int, default=2026)
    p_sim.add_argument("--format", choices=["text", "json"], default="text")
    p_sim.set_defaults(func=_cmd_simulate)

    def _add_model_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", required=True, help="Input CSV dataset path.")
        p.add_argument("--outcome", default="PV{k}MATH", help="PV column pattern.")
        p.add_argument("--design", default="pisa", choices=["pisa", "timss", "pirls"])
        p.add_argument("--n-reps", type=int, default=40)
        p.add_argument("--n-pvs", type=int, default=5)
        p.add_argument("--system-col", default="CNT")
        p.add_argument("--format", choices=["text", "json"], default="text")

    p_ols = sub.add_parser("ols", help="Fit weighted PV OLS regression.")
    _add_model_args(p_ols)
    p_ols.add_argument("--predictors", required=True, help="Comma-separated predictor names.")
    p_ols.add_argument("--system-fe", action="store_true")
    p_ols.add_argument("--senate", action="store_true")
    p_ols.set_defaults(func=_cmd_ols)

    p_meta = sub.add_parser("meta", help="Pool a regression slope across systems.")
    _add_model_args(p_meta)
    p_meta.add_argument("--predictors", required=True)
    p_meta.add_argument("--term", required=True)
    p_meta.set_defaults(func=_cmd_meta)

    p_ovb = sub.add_parser("sensitivity", help="Run Cinelli-Hazlett OVB sensitivity analysis.")
    _add_model_args(p_ovb)
    p_ovb.add_argument("--treatment", required=True)
    p_ovb.add_argument("--controls", default="")
    p_ovb.add_argument("--benchmarks", default="")
    p_ovb.add_argument("--system-fe", action="store_true")
    p_ovb.set_defaults(func=_cmd_sensitivity)

    p_bench = sub.add_parser("benchmark", help="Run Monte Carlo 95%% coverage benchmark.")
    p_bench.add_argument("--n-sim", type=int, default=50)
    p_bench.add_argument("--seed", type=int, default=909)
    p_bench.add_argument("--out", default="")
    p_bench.add_argument("--format", choices=["text", "json"], default="text")
    p_bench.set_defaults(func=_cmd_benchmark)

    p_eq = sub.add_parser("equiv", help="Verify numerical parity against SciPy, OECD BRR, and dense REML.")
    p_eq.add_argument("--format", choices=["text", "json"], default="text")
    p_eq.set_defaults(func=_cmd_equiv)

    p_real = sub.add_parser("demo-real", help="Run end-to-end estimators on bundled PISA 2022 data.")
    p_real.add_argument("--format", choices=["text", "json"], default="text")
    p_real.set_defaults(func=_cmd_demo_real)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the pisakit command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
