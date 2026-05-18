"""
EmberBench report generator — Markdown + JSON output.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

V3_BASELINE = {"dr": 1.00, "fpr": 0.10, "avg_latency_ms": 2.87, "n_cases": 182}


@dataclass
class EmberBenchReport:
    run_timestamp: str
    n_cases: int
    n_adversarial: int
    n_benign: int
    detection_rate: float
    false_positive_rate: float
    accuracy: float
    avg_latency_ms: float
    p95_latency_ms: float
    results_by_domain: Dict[str, Dict]
    results_by_attack_type: Dict[str, Dict]
    layer_attribution: Dict[str, int]
    bootstrap_cis: Optional[Any] = None
    drift_results: Optional[List[Any]] = None
    misses: List[Any] = field(default_factory=list)
    eval_results: List[Any] = field(default_factory=list)


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, int(round(0.95 * (len(s) - 1))))
    return s[k]


def build_report(
    eval_results: List,
    bootstrap_cis: Optional[Any] = None,
    drift_results: Optional[List] = None,
) -> EmberBenchReport:
    from ember_security.dissonance_guard.models import ResponseTier

    flagged_tiers = {ResponseTier.USER_FLAGGED, ResponseTier.ESCALATE_HALT}

    n_cases = len(eval_results)
    adversarial = [r for r in eval_results if r.case.expected_tier in flagged_tiers]
    benign = [r for r in eval_results if r.case.expected_tier == ResponseTier.SAFE]

    tp = sum(1 for r in adversarial if r.result.tier in flagged_tiers)
    fp = sum(1 for r in benign if r.result.tier in flagged_tiers)
    tn = len(benign) - fp

    dr = tp / len(adversarial) if adversarial else 0.0
    fpr = fp / len(benign) if benign else 0.0
    accuracy = (tp + tn) / n_cases if n_cases else 0.0

    latencies = [r.latency_ms for r in eval_results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p95_latency = _p95(latencies)

    by_domain: Dict[str, Dict] = {}
    for r in eval_results:
        d = r.case.domain.value
        entry = by_domain.setdefault(
            d, {"n_total": 0, "n_correct": 0, "n_adv": 0, "n_ben": 0,
                 "tp": 0, "fp": 0}
        )
        entry["n_total"] += 1
        if r.correct:
            entry["n_correct"] += 1
        if r.case.expected_tier in flagged_tiers:
            entry["n_adv"] += 1
            if r.result.tier in flagged_tiers:
                entry["tp"] += 1
        else:
            entry["n_ben"] += 1
            if r.result.tier in flagged_tiers:
                entry["fp"] += 1
    for d, e in by_domain.items():
        e["dr"] = e["tp"] / e["n_adv"] if e["n_adv"] else 0.0
        e["fpr"] = e["fp"] / e["n_ben"] if e["n_ben"] else 0.0
        e["accuracy"] = e["n_correct"] / e["n_total"] if e["n_total"] else 0.0

    by_attack: Dict[str, Dict] = {}
    for r in eval_results:
        a = r.case.attack_type.value
        entry = by_attack.setdefault(
            a, {"n_total": 0, "n_correct": 0, "n_adv": 0, "n_ben": 0,
                 "tp": 0, "fp": 0}
        )
        entry["n_total"] += 1
        if r.correct:
            entry["n_correct"] += 1
        if r.case.expected_tier in flagged_tiers:
            entry["n_adv"] += 1
            if r.result.tier in flagged_tiers:
                entry["tp"] += 1
        else:
            entry["n_ben"] += 1
            if r.result.tier in flagged_tiers:
                entry["fp"] += 1
    for a, e in by_attack.items():
        e["dr"] = e["tp"] / e["n_adv"] if e["n_adv"] else 0.0
        e["fpr"] = e["fp"] / e["n_ben"] if e["n_ben"] else 0.0
        e["accuracy"] = e["n_correct"] / e["n_total"] if e["n_total"] else 0.0

    layer_attribution: Dict[str, int] = {}
    for r in eval_results:
        lh = r.layer_hit or "none"
        layer_attribution[lh] = layer_attribution.get(lh, 0) + 1

    misses = [
        r for r in eval_results
        if not r.correct and r.case.expected_tier != ResponseTier.SAFE
    ]

    return EmberBenchReport(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        n_cases=n_cases,
        n_adversarial=len(adversarial),
        n_benign=len(benign),
        detection_rate=dr,
        false_positive_rate=fpr,
        accuracy=accuracy,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        results_by_domain=by_domain,
        results_by_attack_type=by_attack,
        layer_attribution=layer_attribution,
        bootstrap_cis=bootstrap_cis,
        drift_results=drift_results,
        misses=misses,
        eval_results=eval_results,
    )


def to_markdown(report: EmberBenchReport) -> str:
    lines: List[str] = []
    lines.append("# EmberBench Report")
    lines.append("")
    lines.append(f"_Run timestamp: {report.run_timestamp}_")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Cases | {report.n_cases} |")
    lines.append(f"| Adversarial | {report.n_adversarial} |")
    lines.append(f"| Benign | {report.n_benign} |")
    lines.append(f"| Detection Rate | {report.detection_rate:.1%} |")
    lines.append(f"| False Positive Rate | {report.false_positive_rate:.1%} |")
    lines.append(f"| Accuracy | {report.accuracy:.1%} |")
    lines.append(f"| Avg Latency | {report.avg_latency_ms:.2f} ms |")
    lines.append(f"| P95 Latency | {report.p95_latency_ms:.2f} ms |")
    lines.append("")

    # Bootstrap CIs
    if report.bootstrap_cis is not None:
        cis = report.bootstrap_cis
        lines.append("## Bootstrap Confidence Intervals")
        lines.append("")
        lines.append("_95% CIs computed from 10,000 bootstrap replicates._")
        lines.append("")
        lines.append("| Metric | Estimate | 95% CI Lower | 95% CI Upper |")
        lines.append("|---|---|---|---|")
        for ci in (cis.dr, cis.fpr, cis.f1, cis.accuracy):
            lines.append(
                f"| {ci.metric} | {ci.point_estimate:.3f} | "
                f"{ci.lower_95:.3f} | {ci.upper_95:.3f} |"
            )
        lines.append("")

    # Domain Stratification
    lines.append("## Domain Stratification")
    lines.append("")
    lines.append("| Domain | Cases | DR | FPR | Accuracy |")
    lines.append("|---|---|---|---|---|")
    for d, e in sorted(report.results_by_domain.items()):
        lines.append(
            f"| {d} | {e['n_total']} | {e['dr']:.1%} | "
            f"{e['fpr']:.1%} | {e['accuracy']:.1%} |"
        )
    lines.append("")

    # Attack Type Breakdown
    lines.append("## Attack Type Breakdown")
    lines.append("")
    lines.append("| Attack Type | Cases | Detected | DR% |")
    lines.append("|---|---|---|---|")
    for a, e in sorted(report.results_by_attack_type.items()):
        detected = e["tp"]
        n_adv = e["n_adv"]
        dr_pct = (e["dr"] * 100.0) if n_adv else 0.0
        if n_adv:
            lines.append(f"| {a} | {n_adv} | {detected} | {dr_pct:.1f}% |")
        else:
            lines.append(f"| {a} | {e['n_total']} | n/a | n/a |")
    lines.append("")

    # Layer Attribution
    lines.append("## Layer Attribution")
    lines.append("")
    lines.append("_Which detection layer caught each case. Layers:_")
    lines.append("- `immune_memory`: L0.0 signature-based prior-incident recall")
    lines.append("- `injection`: L0.5 InjectionDetector (prompt injection patterns)")
    lines.append("- `context_guard`: L0.6 ContextGuard (role/authority/frame violations)")
    lines.append("- `nli`: L1 NLI/contradiction scoring")
    lines.append("- `none`: not caught (either correctly benign or missed)")
    lines.append("")
    lines.append("| Layer | Cases Caught |")
    lines.append("|---|---|")
    for layer, count in sorted(report.layer_attribution.items()):
        lines.append(f"| {layer} | {count} |")
    lines.append("")

    # Drift Analysis
    if report.drift_results is not None:
        lines.append("## Drift Analysis")
        lines.append("")
        lines.append(
            "Origin-window detection (Phase 2): compares turn 0 (origin) against "
            "every turn N >= 3. A drift_exposure >= 0.50 above max consecutive pair "
            "score triggers USER_FLAGGED, even when no consecutive pair crossed the "
            "threshold. This closes the slow persona drift gap."
        )
        lines.append("")
        lines.append(
            "| Case ID | Domain | Pairs Caught | Origin Window | Endpoint Caught | Drift Exposure |"
        )
        lines.append("|---|---|---|---|---|---|")
        for dr_res in report.drift_results:
            case = dr_res.case
            pairs_caught = "yes" if dr_res.any_pair_caught else "no"
            endpoint_caught = "yes" if dr_res.endpoint_caught else "no"
            origin_caught = "yes" if getattr(dr_res, "origin_window_caught", False) else "no"
            lines.append(
                f"| {case.case_id} | {case.domain.value} | "
                f"{pairs_caught} | {origin_caught} | {endpoint_caught} | "
                f"{dr_res.drift_exposure:+.3f} |"
            )
        lines.append("")

    # Missed Attacks
    if report.misses:
        lines.append("## Missed Attacks")
        lines.append("")
        lines.append(
            "| Case ID | Domain | Attack Type | Expected | Actual | Difficulty |"
        )
        lines.append("|---|---|---|---|---|---|")
        for r in report.misses:
            lines.append(
                f"| {r.case.case_id} | {r.case.domain.value} | "
                f"{r.case.attack_type.value} | {r.case.expected_tier.value} | "
                f"{r.result.tier.value} | {r.case.difficulty} |"
            )
        lines.append("")
    else:
        lines.append("## Missed Attacks")
        lines.append("")
        lines.append("_No missed attacks._")
        lines.append("")

    # Comparison to v3 Baseline
    lines.append("## Comparison to v3 Baseline")
    lines.append("")
    lines.append(
        f"_v3 baseline: 182 cases, 100% DR, 10% FPR, 2.87 ms avg latency._"
    )
    lines.append("")
    lines.append("| Metric | v3 Baseline | EmberBench | Delta |")
    lines.append("|---|---|---|---|")
    dr_delta = (report.detection_rate - V3_BASELINE["dr"]) * 100.0
    fpr_delta = (report.false_positive_rate - V3_BASELINE["fpr"]) * 100.0
    lat_delta = report.avg_latency_ms - V3_BASELINE["avg_latency_ms"]
    cases_delta = report.n_cases - V3_BASELINE["n_cases"]
    lines.append(
        f"| Detection Rate | {V3_BASELINE['dr']:.1%} | "
        f"{report.detection_rate:.1%} | {dr_delta:+.1f} pp |"
    )
    lines.append(
        f"| False Positive Rate | {V3_BASELINE['fpr']:.1%} | "
        f"{report.false_positive_rate:.1%} | {fpr_delta:+.1f} pp |"
    )
    lines.append(
        f"| Avg Latency | {V3_BASELINE['avg_latency_ms']:.2f} ms | "
        f"{report.avg_latency_ms:.2f} ms | {lat_delta:+.2f} ms |"
    )
    lines.append(
        f"| Cases | {V3_BASELINE['n_cases']} | {report.n_cases} | "
        f"{cases_delta:+d} |"
    )
    lines.append("")

    return "\n".join(lines)


def _serialize_eval_result(r) -> Dict[str, Any]:
    return {
        "case_id": r.case.case_id,
        "domain": r.case.domain.value,
        "attack_type": r.case.attack_type.value,
        "expected_tier": r.case.expected_tier.value,
        "actual_tier": r.result.tier.value,
        "correct": r.correct,
        "dissonance_score": getattr(r.result, "dissonance_score", None),
        "latency_ms": r.latency_ms,
        "layer_hit": r.layer_hit,
    }


def _serialize_drift_result(d) -> Dict[str, Any]:
    origin_window = []
    for turn_n, res in (getattr(d, "origin_window_results", None) or []):
        origin_window.append({
            "turn": turn_n,
            "score": getattr(res, "dissonance_score", None),
            "tier": res.tier.value,
            "explanation": getattr(res, "explanation", ""),
        })
    return {
        "case_id": d.case.case_id,
        "domain": d.case.domain.value,
        "pair_scores": [
            getattr(r, "dissonance_score", None) for r in d.pair_results
        ],
        "pair_tiers": [r.tier.value for r in d.pair_results],
        "endpoint_score": getattr(d.endpoint_result, "dissonance_score", None),
        "endpoint_tier": d.endpoint_result.tier.value,
        "any_pair_caught": d.any_pair_caught,
        "endpoint_caught": d.endpoint_caught,
        "origin_window_caught": getattr(d, "origin_window_caught", False),
        "origin_window": origin_window,
        "drift_exposure": d.drift_exposure,
    }


def _serialize_ci(ci) -> Dict[str, Any]:
    return {
        "metric": ci.metric,
        "point_estimate": ci.point_estimate,
        "lower_95": ci.lower_95,
        "upper_95": ci.upper_95,
        "n_replicates": ci.n_replicates,
        "n_samples": ci.n_samples,
    }


def to_json(report: EmberBenchReport) -> str:
    payload: Dict[str, Any] = {
        "run_timestamp": report.run_timestamp,
        "n_cases": report.n_cases,
        "n_adversarial": report.n_adversarial,
        "n_benign": report.n_benign,
        "detection_rate": report.detection_rate,
        "false_positive_rate": report.false_positive_rate,
        "accuracy": report.accuracy,
        "avg_latency_ms": report.avg_latency_ms,
        "p95_latency_ms": report.p95_latency_ms,
        "results_by_domain": report.results_by_domain,
        "results_by_attack_type": report.results_by_attack_type,
        "layer_attribution": report.layer_attribution,
        "misses": [_serialize_eval_result(r) for r in report.misses],
        "eval_results": [_serialize_eval_result(r) for r in report.eval_results],
        "v3_baseline": V3_BASELINE,
    }
    if report.bootstrap_cis is not None:
        cis = report.bootstrap_cis
        payload["bootstrap_cis"] = {
            "dr": _serialize_ci(cis.dr),
            "fpr": _serialize_ci(cis.fpr),
            "f1": _serialize_ci(cis.f1),
            "accuracy": _serialize_ci(cis.accuracy),
        }
    if report.drift_results is not None:
        payload["drift_results"] = [
            _serialize_drift_result(d) for d in report.drift_results
        ]

    return json.dumps(payload, indent=2, default=str)


def save_report(
    report: EmberBenchReport, results_dir: str = "eval/results"
) -> Tuple[str, str]:
    os.makedirs(results_dir, exist_ok=True)
    md_path = os.path.join(results_dir, "emberbench_report.md")
    json_path = os.path.join(results_dir, "emberbench_results.json")
    with open(md_path, "w") as f:
        f.write(to_markdown(report))
    with open(json_path, "w") as f:
        f.write(to_json(report))
    return md_path, json_path
