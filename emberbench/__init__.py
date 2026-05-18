"""EmberBench — adversarial benchmark for the EmberArmor pipeline."""
from .runner import DriftResult, EmberBenchRunner
from .datasets.base import AttackType, Domain, DriftCase
from .bootstrap import BootstrapCI, EmberBenchCIs, compute_bootstrap_cis
from .report import EmberBenchReport, build_report, save_report, to_json, to_markdown

__all__ = [
    "AttackType",
    "BootstrapCI",
    "Domain",
    "DriftCase",
    "DriftResult",
    "EmberBenchCIs",
    "EmberBenchReport",
    "EmberBenchRunner",
    "build_report",
    "compute_bootstrap_cis",
    "save_report",
    "to_json",
    "to_markdown",
]
