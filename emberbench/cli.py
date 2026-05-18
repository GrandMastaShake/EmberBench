"""
EmberBench CLI.

Usage:
    python -m eval.emberbench                                # full run
    python -m eval.emberbench --fast                         # L0.5/L0.6 only
    python -m eval.emberbench --domain legal                 # single domain
    python -m eval.emberbench --drift                        # drift sequences only
    python -m eval.emberbench --bootstrap                    # with 10K bootstrap CIs
    python -m eval.emberbench --save                         # save to eval/results/
    python -m eval.emberbench --compare-baseline             # vs v3 baseline
    python -m eval.emberbench --fast --domain financial --save
"""
from __future__ import annotations

import argparse
import sys
from typing import List


def _print_baseline_delta(report) -> None:
    from eval.emberbench.report import V3_BASELINE

    dr_delta_pp = (report.detection_rate - V3_BASELINE["dr"]) * 100.0
    fpr_delta_pp = (report.false_positive_rate - V3_BASELINE["fpr"]) * 100.0
    lat_delta = report.avg_latency_ms - V3_BASELINE["avg_latency_ms"]

    print()
    print("── vs v3 Baseline (182 cases, public benchmarks) ──────────────────")
    print(f"  {'Metric':18s}{'v3 Baseline':15s}{'EmberBench':15s}{'Delta'}")
    print(
        f"  {'Detection Rate':18s}"
        f"{V3_BASELINE['dr'] * 100:<15.1f}"
        f"{report.detection_rate * 100:<15.1f}"
        f"{dr_delta_pp:+.1f} pp"
    )
    print(
        f"  {'False Pos Rate':18s}"
        f"{V3_BASELINE['fpr'] * 100:<15.1f}"
        f"{report.false_positive_rate * 100:<15.1f}"
        f"{fpr_delta_pp:+.1f} pp"
    )
    print(
        f"  {'Avg Latency':18s}"
        f"{V3_BASELINE['avg_latency_ms']:<15.2f}"
        f"{report.avg_latency_ms:<15.2f}"
        f"{lat_delta:+.2f}ms"
    )
    print("─────────────────────────────────────────────────────────────────────")


def _print_drift_summary(drift_results: List) -> None:
    if not drift_results:
        print("No drift cases.")
        return
    n = len(drift_results)
    n_pairs_caught = sum(1 for d in drift_results if d.any_pair_caught)
    n_endpoint_caught = sum(1 for d in drift_results if d.endpoint_caught)
    avg_exposure = sum(d.drift_exposure for d in drift_results) / n

    print()
    print("── Drift Sequence Analysis ──────────────────────────────────────────")
    print(f"  Total drift cases:         {n}")
    print(f"  Any consecutive pair hit:  {n_pairs_caught}/{n}")
    print(f"  Endpoint pair hit:         {n_endpoint_caught}/{n}")
    print(f"  Avg drift exposure:        {avg_exposure:+.3f}")
    print()
    print(f"  {'case_id':24s} {'domain':10s} {'pairs':>6s} {'endpoint':>9s} {'exposure':>10s}")
    for d in drift_results:
        pc = "yes" if d.any_pair_caught else "no"
        ec = "yes" if d.endpoint_caught else "no"
        print(
            f"  {d.case.case_id:24s} {d.case.domain.value:10s} "
            f"{pc:>6s} {ec:>9s} {d.drift_exposure:+10.3f}"
        )
    print("─────────────────────────────────────────────────────────────────────")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="EmberBench — adversarial benchmark CLI."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip NLI; test L0.5/L0.6 pattern layers only",
    )
    parser.add_argument(
        "--domain",
        choices=["legal", "financial", "medical", "drift", "all"],
        default="all",
    )
    parser.add_argument(
        "--drift", action="store_true",
        help="Run drift sequences only (no single-pair eval)",
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Compute 10K bootstrap CIs (adds 30-60s)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save report to eval/results/",
    )
    parser.add_argument(
        "--compare-baseline", action="store_true",
        help="Print comparison vs v3 baseline",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-case results",
    )

    args = parser.parse_args()

    from eval.emberbench.runner import EmberBenchRunner
    from eval.emberbench.report import build_report, save_report
    from eval.emberbench.bootstrap import compute_bootstrap_cis, print_bootstrap_report
    from eval.emberbench.datasets import get_all_cases
    from eval.emberbench.datasets.base import Domain
    from eval.run_eval_direct import run_eval, print_summary

    # Drift-only path
    if args.drift or args.domain == "drift":
        runner = EmberBenchRunner(fast=args.fast, seed=args.seed)
        drift_results = runner.run_drift()
        _print_drift_summary(drift_results)
        if args.save:
            # Build a minimal report with drift_results and save
            report = build_report([], drift_results=drift_results)
            md_path, json_path = save_report(report)
            print(f"\nSaved: {md_path}")
            print(f"Saved: {json_path}")
        return 0

    # Filter cases by domain if necessary
    cases = get_all_cases()
    if args.domain != "all":
        target = Domain(args.domain)
        cases = [c for c in cases if c.domain == target]

    if not cases:
        print("No cases matched the filter.", file=sys.stderr)
        return 2

    runner = EmberBenchRunner(fast=args.fast, seed=args.seed)
    eval_results = run_eval(cases, fast=args.fast)

    drift_results = None
    if not args.fast:
        drift_results = runner.run_drift()

    cis = None
    if args.bootstrap:
        cis = compute_bootstrap_cis(eval_results, seed=args.seed)

    full_report = build_report(eval_results, cis, drift_results)

    print_summary(eval_results)

    if args.verbose:
        print("\nPer-case results:")
        for r in eval_results:
            status = "OK " if r.correct else "MISS"
            print(
                f"  [{status}] {r.case.case_id:24s} "
                f"expected={r.case.expected_tier.value:14s} "
                f"actual={r.result.tier.value:14s} "
                f"lat={r.latency_ms:6.2f}ms"
            )

    if args.bootstrap and cis is not None:
        print_bootstrap_report(cis)

    if drift_results is not None:
        _print_drift_summary(drift_results)

    if args.compare_baseline:
        _print_baseline_delta(full_report)

    if args.save:
        md_path, json_path = save_report(full_report)
        print(f"\nSaved: {md_path}")
        print(f"Saved: {json_path}")

    return 1 if full_report.detection_rate < 0.80 else 0


if __name__ == "__main__":
    sys.exit(main() or 0)
