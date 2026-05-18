from .models import ResponseTier
from ..config import DissonanceConfig


def route_tier(dissonance_score: float, config: DissonanceConfig) -> ResponseTier:
    """Map dissonance score → response tier."""
    if dissonance_score < config.auto_resolve_threshold:
        return ResponseTier.SAFE
    elif dissonance_score < config.user_flag_threshold:
        return ResponseTier.AUTO_RESOLVE
    elif dissonance_score < config.halt_threshold:
        return ResponseTier.USER_FLAGGED
    else:
        return ResponseTier.ESCALATE_HALT
