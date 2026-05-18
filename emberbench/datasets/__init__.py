"""EmberBench adversarial datasets."""
from .legal import get_legal_cases
from .financial import get_financial_cases
from .medical import get_medical_cases
from .drift import get_drift_cases


def get_all_cases():
    from eval.run_eval_direct import EvalCase
    cases: list[EvalCase] = []
    cases.extend(get_legal_cases())
    cases.extend(get_financial_cases())
    cases.extend(get_medical_cases())
    return cases


def get_all_drift_cases():
    return get_drift_cases()


__all__ = [
    "get_legal_cases",
    "get_financial_cases",
    "get_medical_cases",
    "get_drift_cases",
    "get_all_cases",
    "get_all_drift_cases",
]
