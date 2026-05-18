"""
Bootstrap CI for EmberBench metrics (10,000 replicates default).

Computes 95% confidence intervals for:
  - Detection Rate (DR)
  - False Positive Rate (FPR)
  - F1 score
  - Accuracy
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BootstrapCI:
    metric: str
    point_estimate: float
    lower_95: float
    upper_95: float
    n_replicates: int
    n_samples: int

    def __str__(self) -> str:
        return (
            f"{self.metric}: {self.point_estimate:.3f} "
            f"[{self.lower_95:.3f}, {self.upper_95:.3f}] "
            f"(95% CI, {self.n_replicates:,} replicates)"
        )


@dataclass
class EmberBenchCIs:
    dr: BootstrapCI
    fpr: BootstrapCI
    f1: BootstrapCI
    accuracy: BootstrapCI


def compute_bootstrap_cis(
    results: List,
    n_replicates: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> EmberBenchCIs:
    """
    Compute bootstrap CIs for DR, FPR, F1, accuracy.

    Adversarial = expected_tier in {USER_FLAGGED, ESCALATE_HALT}
    Benign = expected_tier == SAFE
    """
    from ember_security.dissonance_guard.models import ResponseTier

    rng = np.random.default_rng(seed)
    flagged_tiers = {ResponseTier.USER_FLAGGED, ResponseTier.ESCALATE_HALT}

    adversarial = [r for r in results if r.case.expected_tier in flagged_tiers]
    benign = [r for r in results if r.case.expected_tier == ResponseTier.SAFE]

    def _compute_metrics(adv_sample, ben_sample):
        tp = sum(1 for r in adv_sample if r.result.tier in flagged_tiers)
        fp = sum(1 for r in ben_sample if r.result.tier in flagged_tiers)
        tn = len(ben_sample) - fp
        fn = len(adv_sample) - tp

        dr = tp / len(adv_sample) if adv_sample else 0.0
        fpr = fp / len(ben_sample) if ben_sample else 0.0
        precision = 1 - fpr
        f1 = (2 * dr * precision / (dr + precision)) if (dr + precision) > 0 else 0.0
        acc = (
            (tp + tn) / (len(adv_sample) + len(ben_sample))
            if (adv_sample or ben_sample)
            else 0.0
        )
        return dr, fpr, f1, acc

    point_dr, point_fpr, point_f1, point_acc = _compute_metrics(adversarial, benign)

    dr_samples, fpr_samples, f1_samples, acc_samples = [], [], [], []
    adv_arr = np.array(adversarial, dtype=object)
    ben_arr = np.array(benign, dtype=object)

    for _ in range(n_replicates):
        if len(adv_arr) > 0:
            adv_idx = rng.integers(0, len(adv_arr), size=len(adv_arr))
            adv_s = adv_arr[adv_idx].tolist()
        else:
            adv_s = []
        if len(ben_arr) > 0:
            ben_idx = rng.integers(0, len(ben_arr), size=len(ben_arr))
            ben_s = ben_arr[ben_idx].tolist()
        else:
            ben_s = []
        dr, fpr, f1, acc = _compute_metrics(adv_s, ben_s)
        dr_samples.append(dr)
        fpr_samples.append(fpr)
        f1_samples.append(f1)
        acc_samples.append(acc)

    lo, hi = (alpha / 2) * 100, (1 - alpha / 2) * 100

    def _ci(metric, point, samples, n):
        return BootstrapCI(
            metric=metric,
            point_estimate=point,
            lower_95=float(np.percentile(samples, lo)),
            upper_95=float(np.percentile(samples, hi)),
            n_replicates=n_replicates,
            n_samples=n,
        )

    return EmberBenchCIs(
        dr=_ci("Detection Rate", point_dr, dr_samples, len(adversarial)),
        fpr=_ci("False Positive Rate", point_fpr, fpr_samples, len(benign)),
        f1=_ci("F1 Score", point_f1, f1_samples, len(adversarial) + len(benign)),
        accuracy=_ci("Accuracy", point_acc, acc_samples, len(results)),
    )


def print_bootstrap_report(cis: EmberBenchCIs) -> None:
    print("\n── Bootstrap CIs (95%, 10K replicates) ─────────────────────────────")
    print(f"  {cis.dr}")
    print(f"  {cis.fpr}")
    print(f"  {cis.f1}")
    print(f"  {cis.accuracy}")
    print("─────────────────────────────────────────────────────────────────────")
