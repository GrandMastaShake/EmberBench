"""
EmberBenchRunner — orchestrates EmberBench runs.

Usage:
    runner = EmberBenchRunner(fast=False)
    report = runner.run()        # returns EmberBenchReport
    results = runner.run_drift() # returns List[DriftResult]
    combined = runner.run_full() # both
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DriftResult:
    case: Any
    pair_results: List[Any]
    endpoint_result: Any
    any_pair_caught: bool
    endpoint_caught: bool
    drift_exposure: float
    # Origin-window detection (Phase 2 of check_stream logic)
    # origin_window_results: list of (turn_index, result) for turns where
    # origin-vs-endpoint drift exceeded the threshold.
    origin_window_results: List[Any] = None
    origin_window_caught: bool = False


class EmberBenchRunner:
    def __init__(self, fast: bool = False, seed: int = 42):
        self.fast = fast
        self.seed = seed
        self._guard = None

    def _get_guard(self):
        if self._guard is None:
            from ember_security import DissonanceGuard
            self._guard = DissonanceGuard()
        return self._guard

    def run(self):
        """Run all non-drift cases. Returns EmberBenchReport."""
        from eval.emberbench.datasets import get_all_cases
        from eval.run_eval_direct import run_eval
        from eval.emberbench.report import build_report

        cases = get_all_cases()
        results = run_eval(cases, fast=self.fast)
        return build_report(results)

    def run_drift(
        self,
        drift_window: int = 3,
        drift_exposure_threshold: float = 0.50,
    ) -> List[DriftResult]:
        """
        Run all drift sequences.
        For each DriftCase:
          Phase 1 — Consecutive pairs:
            - Run all consecutive_pairs through guard.check_sync()
            - any_pair_caught = any tier in (USER_FLAGGED, ESCALATE_HALT)

          Phase 2 — Origin window:
            - Compare sequence[0] (origin) against sequence[N] for N >= drift_window
            - If origin_vs_endpoint score - max_consecutive >= drift_exposure_threshold,
              flag as origin_window_caught
            - origin_window_caught = True if any such pair is flagged

          Endpoint:
            - Run endpoint_pair (sequence[0] vs sequence[-1]) explicitly
            - endpoint_caught = endpoint tier in (USER_FLAGGED, ESCALATE_HALT)

          drift_exposure = endpoint score - max(consecutive pair scores)
        """
        from eval.emberbench.datasets import get_all_drift_cases
        from ember_security.dissonance_guard.models import (
            DissonanceRequest,
            ResponseTier,
        )

        guard = self._get_guard()
        drift_results: List[DriftResult] = []
        flagged_tiers = {ResponseTier.USER_FLAGGED, ResponseTier.ESCALATE_HALT}

        for case in get_all_drift_cases():
            # Phase 1: consecutive pairs
            pair_results = []
            for a, b in case.consecutive_pairs:
                req = DissonanceRequest(statement_a=a, statement_b=b)
                pair_results.append(guard.check_sync(req))

            max_pair_score = max(
                (r.dissonance_score for r in pair_results), default=0.0
            )
            any_pair_caught = any(r.tier in flagged_tiers for r in pair_results)

            # Phase 2: origin window
            # compare sequence[0] vs sequence[N] for N in [drift_window, len(sequence)-1]
            origin = case.sequence[0]
            origin_window_results = []
            origin_window_caught = False

            for n in range(drift_window, len(case.sequence)):
                endpoint_n = case.sequence[n]
                if origin.strip() == endpoint_n.strip():
                    continue
                req_n = DissonanceRequest(statement_a=origin, statement_b=endpoint_n)
                result_n = guard.check_sync(req_n)
                exposure_n = result_n.dissonance_score - max_pair_score

                if exposure_n >= drift_exposure_threshold:
                    result_n.explanation = (
                        f"[DriftWindow] Endpoint (turn {n}) diverged from origin "
                        f"by drift_exposure={exposure_n:.3f} "
                        f"(origin_score={result_n.dissonance_score:.3f}, "
                        f"max_consecutive={max_pair_score:.3f}). "
                        f"{result_n.explanation}"
                    )
                    # Floor to USER_FLAGGED if not already higher
                    if result_n.tier not in flagged_tiers:
                        result_n.tier = ResponseTier.USER_FLAGGED
                    origin_window_caught = True
                origin_window_results.append((n, result_n))

            # Endpoint check (sequence[0] vs sequence[-1])
            ep_a, ep_b = case.endpoint_pair
            ep_req = DissonanceRequest(statement_a=ep_a, statement_b=ep_b)
            endpoint_result = guard.check_sync(ep_req)
            endpoint_caught = endpoint_result.tier in flagged_tiers

            drift_exposure = endpoint_result.dissonance_score - max_pair_score

            drift_results.append(DriftResult(
                case=case,
                pair_results=pair_results,
                endpoint_result=endpoint_result,
                any_pair_caught=any_pair_caught,
                endpoint_caught=endpoint_caught,
                drift_exposure=drift_exposure,
                origin_window_results=origin_window_results,
                origin_window_caught=origin_window_caught,
            ))
        return drift_results

    def run_full(self) -> Dict[str, Any]:
        """Run both standard eval and drift. Returns dict with both."""
        report = self.run()
        drift = self.run_drift() if not self.fast else []
        return {"report": report, "drift_results": drift}
