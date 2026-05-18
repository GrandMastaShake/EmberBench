"""Base types for EmberBench datasets."""
from dataclasses import dataclass
from enum import Enum
from typing import List


class Domain(str, Enum):
    LEGAL = "legal"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    DRIFT = "drift"
    GENERAL = "general"


class AttackType(str, Enum):
    SEMANTIC_PARAPHRASE = "semantic_paraphrase"
    AUTHORITY_POISON = "authority_poison"
    NUMERIC_CONTRADICTION = "numeric_contradiction"
    TEMPORAL_INJECTION = "temporal_injection"
    SOFT_INJECTION = "soft_injection"
    CROSS_LAYER_GAP = "cross_layer_gap"
    FRAME_COLLISION_EDGE = "frame_collision_edge"
    DIALOGUE_DRIFT = "dialogue_drift"
    BENIGN_FPR = "benign_fpr"


class DifficultyLevel(int, Enum):
    TRIVIAL = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    ADVERSARIAL = 5


@dataclass
class DriftCase:
    case_id: str
    sequence: List[str]
    domain: Domain
    title: str = ""
    notes: str = ""

    @property
    def consecutive_pairs(self):
        return [(self.sequence[i], self.sequence[i + 1]) for i in range(len(self.sequence) - 1)]

    @property
    def endpoint_pair(self):
        """First vs last — the gap that drift testing exposes."""
        return (self.sequence[0], self.sequence[-1])
