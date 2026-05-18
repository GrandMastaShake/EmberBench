from enum import Enum
from typing import Any, List, Optional

import time

from pydantic import BaseModel, Field


class ResponseTier(str, Enum):
    SAFE = "SAFE"
    AUTO_RESOLVE = "AUTO_RESOLVE"
    USER_FLAGGED = "USER_FLAGGED"
    ESCALATE_HALT = "ESCALATE_HALT"


class DissonanceRequest(BaseModel):
    statement_a: str = Field(..., description="First AI output or claim")
    statement_b: str = Field(..., description="Second AI output or claim to check against A")
    context: Optional[str] = Field(None, description="Optional context window")
    agent_id: Optional[str] = Field(None, description="Source agent identifier")
    session_id: Optional[str] = Field(None)


class DissonanceResult(BaseModel):
    tier: ResponseTier
    dissonance_score: float = Field(..., ge=0.0, le=1.0)
    spatial_similarity: float = Field(..., ge=0.0, le=1.0)
    harmonic_coherence: float = Field(..., ge=0.0, le=1.0)
    contradiction_probability: float = Field(..., ge=0.0, le=1.0)
    entailment_probability: float = Field(..., ge=0.0, le=1.0)
    neutral_probability: float = Field(..., ge=0.0, le=1.0)
    latency_ms: float
    explanation: str
    timestamp: float = Field(default_factory=time.time)
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    sonar_context: Optional[Any] = Field(
        None,
        description="Real-world incident context from Perplexity Sonar (attached on FLAG/HALT)",
    )
    frame_collision: Optional[dict] = Field(
        None,
        description="FrameCollisionResult.as_dict() if collision detected",
    )
    emberscan_result: Optional[Any] = Field(
        None,
        description="EmberScanResult from EmberSCAN Stage 1 (when EMBER_EMBERSCAN_ENABLED=true)",
    )
    freshness_gate_result: Optional[Any] = Field(
        None,
        description="FreshnessGateResult from Zone 3 Freshness Gate (when EMBER_FRESHNESS_GATE_ENABLED=true)",
    )
    temporal_contradictions: Optional[List[Any]] = Field(
        None,
        description="List of TemporalContradiction from temporal contradiction detection",
    )
    positional_weight: Optional[float] = Field(
        None,
        description="w_positional from U-shape attention model — 1.0 at boundaries, ~0.5 at midpoint. "
        "Populated when position_a, position_b, context_length are provided to check().",
    )

    @property
    def is_safe(self) -> bool:
        return self.tier == ResponseTier.SAFE

    @property
    def should_halt(self) -> bool:
        return self.tier == ResponseTier.ESCALATE_HALT
