"""
DissonanceGuard — core detector.

Uses DeBERTa-v3-large cross-encoder for NLI (contradiction / entailment / neutral).
Sentence-BERT for spatial similarity.
Scorer combines into dissonance score D.
Router maps D → 3-tier response.
"""
import time
import asyncio
from functools import lru_cache
from typing import Optional
import structlog

from ..config import DissonanceConfig
from .frame_annotator import FrameAnnotator, FrameCollisionResult
from .models import DissonanceRequest, DissonanceResult, ResponseTier
from .scorer import compute_dissonance
from .router import route_tier

log = structlog.get_logger(__name__)


@lru_cache(maxsize=1)
def _load_nli_model(model_name: str):
    """Lazy-load NLI cross-encoder (cached after first call)."""
    from sentence_transformers import CrossEncoder
    log.info("loading_nli_model", model=model_name)
    return CrossEncoder(model_name, num_labels=3)


@lru_cache(maxsize=1)
def _load_similarity_model():
    """Lazy-load sentence similarity model."""
    from sentence_transformers import SentenceTransformer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    log.info("loading_similarity_model", model=model_name)
    return SentenceTransformer(model_name)


class DissonanceGuard:
    """
    Real-time contradiction detection engine.

    Usage:
        guard = DissonanceGuard()
        result = await guard.check("The system is stable.", "The system crashed.")
        if result.should_halt:
            raise PipelineHaltError(result.explanation)
    """

    def __init__(self, config: Optional[DissonanceConfig] = None, sonar=None):
        self.config = config or DissonanceConfig()
        self._nli = None
        self._sim = None
        self._frame_annotator = FrameAnnotator()
        # Temporal Grounding Layer — decay + heartbeat + proof of presence
        from ..temporal_grounding import TemporalGroundingLayer
        self._tgl = TemporalGroundingLayer()
        # Optional Sonar enricher — auto-initialized if PERPLEXITY_API_KEY is set
        if sonar is None:
            import os
            if os.environ.get("PERPLEXITY_API_KEY"):
                from ..sonar import SonarContextEnricher
                sonar = SonarContextEnricher()
        self._sonar = sonar

    def _ensure_loaded(self):
        if self._nli is None:
            self._nli = _load_nli_model(self.config.model)
        if self._sim is None:
            self._sim = _load_similarity_model()

    def _compute_spatial_similarity(self, a: str, b: str) -> float:
        import numpy as np
        embeddings = self._sim.encode([a, b], normalize_embeddings=True)
        return float(np.dot(embeddings[0], embeddings[1]))

    def _compute_nli(self, a: str, b: str) -> tuple[float, float, float]:
        """Returns (contradiction, entailment, neutral) probabilities."""
        import torch, numpy as np
        scores = self._nli.predict([(a, b)], apply_softmax=True)
        # DeBERTa NLI label order: contradiction=0, entailment=1, neutral=2
        s = scores[0]
        return float(s[0]), float(s[1]), float(s[2])

    def check_sync(self, request: DissonanceRequest) -> DissonanceResult:
        """Synchronous check — use in non-async contexts.

        Pre-screens against immune memory (zero model cost), then runs full
        NLI + spatial similarity + dissonance computation.
        """
        # Pre-screen against immune memory — zero model cost
        try:
            from ..offensive.signatures import get_registry
            registry = get_registry()
            combined = f"{request.statement_a} {request.statement_b}"
            is_known, sig = registry.is_known_attack(
                combined,
                ip=getattr(request, "agent_id", "") or "",
            )
            if is_known:
                return DissonanceResult(
                    dissonance_score=1.0,
                    spatial_similarity=0.0,
                    harmonic_coherence=0.0,
                    # Required NLI probability fields — zeroed on immune-memory fast path
                    # (no model inference runs; the signature match is definitive)
                    contradiction_probability=1.0,
                    entailment_probability=0.0,
                    neutral_probability=0.0,
                    tier=ResponseTier.ESCALATE_HALT,
                    explanation=f"Known attack signature detected (immune memory). "
                                f"Pattern hash: {sig.sig_hash[:12] if sig else 'unknown'}. "
                                f"Seen {sig.hit_count if sig else '?'} times.",
                    latency_ms=0.0,
                    agent_id=request.agent_id,
                    session_id=request.session_id,
                )
        except Exception:
            pass  # Registry unavailable — proceed to full inference

        # ── Layer 0.5: Prompt Injection & Structural Attack Detector ─────────
        # Zero model cost — runs before NLI. Catches what NLI misses:
        # persona hijacks, HTML injection, privilege escalation tokens,
        # "ignore all previous instructions", system prompt exfiltration.
        try:
            from ..offensive.injection_detector import scan_both, InjectionMatch
            injection = scan_both(request.statement_a, request.statement_b)
            if injection.matched and injection.confidence >= 0.85:
                # High-confidence structural attack — halt immediately, skip NLI
                return DissonanceResult(
                    dissonance_score=min(1.0, 0.80 + injection.dissonance_boost),
                    spatial_similarity=0.0,
                    harmonic_coherence=0.0,
                    contradiction_probability=injection.confidence,
                    entailment_probability=0.0,
                    neutral_probability=1.0 - injection.confidence,
                    tier=ResponseTier.ESCALATE_HALT,
                    explanation=(
                        f"[InjectionDetector] {injection.threat_type.replace('_', ' ').title()} detected "
                        f"via pattern '{injection.pattern_name}'. "
                        f"Matched: '{injection.matched_text[:80]}'. "
                        f"Confidence: {injection.confidence:.0%}. Pipeline halted."
                    ),
                    latency_ms=0.5,
                    agent_id=request.agent_id,
                    session_id=request.session_id,
                )
        except Exception:
            pass  # Injection detector unavailable — continue to NLI

        # ── Layer 0.6: ContextGuard ───────────────────────────
        # Parallel peer to InjectionDetector. Catches what structural patterns miss:
        # PII extraction, authority-claim framing, technical harm lexicon,
        # stale temporal injection. Zero model cost, sub-millisecond.
        try:
            from ..offensive.context_guard import scan_both as cg_scan_both
            context_match = cg_scan_both(request.statement_a, request.statement_b)
            if context_match.matched and context_match.confidence >= 0.85:
                score = min(1.0, 0.78 + context_match.dissonance_boost)
                return DissonanceResult(
                    dissonance_score=score,
                    spatial_similarity=0.0,
                    harmonic_coherence=0.0,
                    contradiction_probability=context_match.confidence,
                    entailment_probability=0.0,
                    neutral_probability=1.0 - context_match.confidence,
                    tier=ResponseTier.ESCALATE_HALT,
                    explanation=(
                        f"[ContextGuard-{context_match.sub_detector}] "
                        f"{context_match.threat_type.replace('_', ' ').title()} detected "
                        f"via pattern '{context_match.pattern_name}'. "
                        f"Matched: '{context_match.matched_text[:80]}'. "
                        f"Confidence: {context_match.confidence:.0%}. Pipeline halted."
                    ),
                    latency_ms=0.5,
                    agent_id=request.agent_id,
                    session_id=request.session_id,
                )
        except Exception:
            pass  # ContextGuard unavailable — continue to NLI

        # ── Layer 0.7: IntentGuard ──────────────────────────────────────────────
        # Detects adversarial intent that is invisible to NLI:
        # - Semantic paraphrase attacks (legal-boilerplate system override pairs
        #   where A and B are CONSISTENT with each other so contra≈0)
        # - Authority poison where A asserts authority and B issues the bypass
        # Sub-E: SystemDirectiveDetector — zero model cost, structural analysis only.
        try:
            from ..offensive.intent_guard import scan_both as ig_scan_both
            intent_match = ig_scan_both(request.statement_a, request.statement_b)
            if intent_match.matched and intent_match.confidence >= 0.85:
                score = min(1.0, 0.78 + intent_match.dissonance_boost)
                return DissonanceResult(
                    dissonance_score=score,
                    spatial_similarity=0.0,
                    harmonic_coherence=0.0,
                    contradiction_probability=intent_match.confidence,
                    entailment_probability=0.0,
                    neutral_probability=1.0 - intent_match.confidence,
                    tier=ResponseTier.ESCALATE_HALT,
                    explanation=(
                        f"[IntentGuard-E] System directive / operational state claim detected "
                        f"via pattern '{intent_match.pattern_name}'. "
                        f"Matched: '{intent_match.matched_text[:80]}'. "
                        f"Confidence: {intent_match.confidence:.0%}. Pipeline halted."
                    ),
                    latency_ms=0.5,
                    agent_id=request.agent_id,
                    session_id=request.session_id,
                )
        except Exception:
            pass  # IntentGuard unavailable — continue to NLI

        self._ensure_loaded()
        t0 = time.perf_counter()

        spatial_similarity = self._compute_spatial_similarity(
            request.statement_a, request.statement_b
        )
        contradiction_prob, entailment_prob, neutral_prob = self._compute_nli(
            request.statement_a, request.statement_b
        )
        dissonance_score, harmonic_coherence, explanation, w_positional = compute_dissonance(
            contradiction_prob, entailment_prob, neutral_prob, spatial_similarity,
            position_a=getattr(request, '_position_a', None),
            position_b=getattr(request, '_position_b', None),
            context_length=getattr(request, '_context_length', None),
            statement_a=request.statement_a,
            statement_b=request.statement_b,
        )
        tier = route_tier(dissonance_score, self.config)

        latency_ms = (time.perf_counter() - t0) * 1000

        if latency_ms > self.config.max_latency_ms:
            log.warning("latency_exceeded", latency_ms=latency_ms, limit=self.config.max_latency_ms)

        result = DissonanceResult(
            tier=tier,
            dissonance_score=dissonance_score,
            spatial_similarity=spatial_similarity,
            harmonic_coherence=harmonic_coherence,
            contradiction_probability=contradiction_prob,
            entailment_probability=entailment_prob,
            neutral_probability=neutral_prob,
            latency_ms=latency_ms,
            explanation=explanation,
            agent_id=request.agent_id,
            session_id=request.session_id,
            positional_weight=w_positional if any(
                getattr(request, attr, None) is not None
                for attr in ('_position_a', '_position_b', '_context_length')
            ) else None,
        )

        # Prometheus metrics — failure here must never break the detection path
        try:
            from ..api.metrics import DISSONANCE_CHECKS_TOTAL, DISSONANCE_LATENCY_MS, DISSONANCE_SCORE
            DISSONANCE_CHECKS_TOTAL.labels(tier=tier.value).inc()
            DISSONANCE_LATENCY_MS.observe(latency_ms)
            DISSONANCE_SCORE.observe(dissonance_score)
        except Exception:
            pass

        log.info(
            "dissonance_check",
            tier=tier.value,
            score=f"{dissonance_score:.3f}",
            latency_ms=f"{latency_ms:.1f}",
            agent_id=request.agent_id,
        )
        return result

    async def check(
        self,
        a: str,
        b: str,
        cited_items=None,
        position_a: int | None = None,
        position_b: int | None = None,
        context_length: int | None = None,
        **kwargs,
    ) -> DissonanceResult:
        """Async check — runs NLI in thread pool, then enriches FLAG/HALT with Sonar context.

        Args:
            position_a: Token/turn position of statement A in context (0-indexed).
            position_b: Token/turn position of statement B in context (0-indexed).
            context_length: Total context length in tokens or turns.
                When all three are provided, w_positional is computed and attached
                to the result as positional_weight.
        """
        # Epistemic Circuit Breaker — if OPEN, reject immediately without inference
        from ..circuit_breaker import get_circuit_breaker, is_circuit_breaker_enabled
        if is_circuit_breaker_enabled() and get_circuit_breaker().is_open():
            cb = get_circuit_breaker()
            return DissonanceResult(
                dissonance_score=1.0,
                spatial_similarity=0.0,
                harmonic_coherence=0.0,
                contradiction_probability=1.0,
                entailment_probability=0.0,
                neutral_probability=0.0,
                tier=ResponseTier.ESCALATE_HALT,
                explanation="CIRCUIT_BREAKER_OPEN: system under adversarial probing",
                latency_ms=0.0,
                agent_id=kwargs.get("agent_id"),
                session_id=kwargs.get("session_id"),
            )

        request = DissonanceRequest(statement_a=a, statement_b=b, **kwargs)
        # Stash positional params on request for check_sync to pick up
        request._position_a = position_a
        request._position_b = position_b
        request._context_length = context_length

        # Pre-scan: if lower-confidence injection signal (0.7–0.84), compound with NLI result
        _injection_boost = 0.0
        _injection_note = ""
        try:
            from ..offensive.injection_detector import scan_both
            inj = scan_both(a, b)
            if inj.matched and 0.70 <= inj.confidence < 0.85:
                _injection_boost = inj.dissonance_boost * 0.5  # partial boost
                _injection_note = (
                    f" [InjectionDetector partial: {inj.threat_type}, "
                    f"conf={inj.confidence:.0%}, +{_injection_boost:.3f}]"
                )
        except Exception:
            pass

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self.check_sync, request)

        # Apply partial injection boost on top of NLI result
        if _injection_boost > 0:
            result.dissonance_score = min(1.0, result.dissonance_score + _injection_boost)
            result.tier = route_tier(result.dissonance_score, self.config)
            result.explanation += _injection_note

        # EmberSCAN Stage 1 pre-scan (opt-in via EMBER_EMBERSCAN_ENABLED env var)
        import os
        if os.environ.get("EMBER_EMBERSCAN_ENABLED", "").lower() == "true":
            try:
                from ..emberscan import EmberScanner
                scanner = EmberScanner()
                context = f"{a} {b}"
                scan_result = scanner.scan(context)
                result.emberscan_result = scan_result
                if scan_result.has_critical:
                    EMBERSCAN_CRITICAL_BOOST = 0.15
                    result.dissonance_score = min(
                        1.0, result.dissonance_score + EMBERSCAN_CRITICAL_BOOST
                    )
                    result.tier = route_tier(result.dissonance_score, self.config)
                    result.explanation = (
                        f"{result.explanation} "
                        f"[EmberSCAN: {scan_result.tier1_count} critical candidate(s) detected]"
                    )
                log.info(
                    "emberscan_stage1_attached",
                    tier1=scan_result.tier1_count,
                    tier2=scan_result.tier2_count,
                    tier3=scan_result.tier3_count,
                    latency_ms=f"{scan_result.scan_latency_ms:.1f}",
                )
            except Exception as e:
                log.warning("emberscan_stage1_skipped", error=str(e))

        # Freshness Gate — Zone 3 enforcement (opt-out via EMBER_FRESHNESS_GATE_ENABLED=false)
        if cited_items is not None and os.environ.get("EMBER_FRESHNESS_GATE_ENABLED", "true").lower() != "false":
            try:
                from ..freshness.gate import FreshnessGate
                from ..freshness.contradiction import TemporalContradictionDetector

                gate = FreshnessGate()
                gate_result = gate.check(cited_items)
                result.freshness_gate_result = gate_result

                detector = TemporalContradictionDetector()
                temporal_contradictions = detector.detect(cited_items)
                result.temporal_contradictions = temporal_contradictions if temporal_contradictions else None

                if not gate_result.passed:
                    result.tier = ResponseTier.ESCALATE_HALT
                    result.explanation = (
                        f"{result.explanation} "
                        f"[Freshness Gate HALT: {gate_result.halt_reason}]"
                    )

                if temporal_contradictions:
                    has_tier1 = any(
                        tc.tier.value == "TIER1_HALT" for tc in temporal_contradictions
                    )
                    if has_tier1 and result.tier not in (ResponseTier.ESCALATE_HALT,):
                        result.tier = ResponseTier.USER_FLAGGED
                        result.explanation = (
                            f"{result.explanation} "
                            f"[Temporal contradiction: TIER1 — upgraded to USER_FLAGGED]"
                        )

                log.info(
                    "freshness_gate_checked",
                    passed=gate_result.passed,
                    tier=gate_result.tier.value,
                    contradictions=len(temporal_contradictions),
                )
            except Exception as e:
                log.warning("freshness_gate_skipped", error=str(e))

        # Temporal grounding — demote stale-domain claims before routing
        decay_a = self._tgl.demote_claim(a)
        decay_b = self._tgl.demote_claim(b)
        max_decay = max(decay_a.decay_applied, decay_b.decay_applied)
        if max_decay > 0.1:  # only adjust for meaningful decay (CVE, financial, regulatory)
            decay_boost = max_decay * 0.3  # conservative: 0.3× the decay applied
            result.dissonance_score = min(1.0, result.dissonance_score + decay_boost)
            result.tier = route_tier(result.dissonance_score, self.config)
            result.explanation = (
                f"{result.explanation} "
                f"[Temporal decay: {decay_a.domain.value}/{decay_b.domain.value}, "
                f"+{decay_boost:.3f}]"
            )

        # Per-session heartbeat turn tracking
        session_id = kwargs.get("session_id", "")
        if session_id:
            session = self._tgl.get_or_create_session(session_id)
            session.heartbeat.advance_turn()

        # Pre-DissonanceGuard frame collision check
        ann_a = self._frame_annotator.annotate(a)
        ann_b = self._frame_annotator.annotate(b)
        frame_result = self._frame_annotator.check_collision(ann_a, ann_b)
        if frame_result.colliding:
            result.frame_collision = frame_result.as_dict()
            # Frame collision → floor dissonance score to USER_FLAGGED tier minimum
            # so that semantically incompatible frames actually affect routing
            FRAME_COLLISION_FLOOR = 0.65
            if result.dissonance_score < FRAME_COLLISION_FLOOR:
                result.dissonance_score = FRAME_COLLISION_FLOOR
                result.tier = route_tier(result.dissonance_score, self.config)
                result.explanation = (
                    f"{result.explanation} "
                    f"[Frame collision: {frame_result.frame_pair[0]} ↔ {frame_result.frame_pair[1]} "
                    f"— score floored to {FRAME_COLLISION_FLOOR:.2f}]"
                )

        # Record HALT attacks to immune memory
        if result.tier == ResponseTier.ESCALATE_HALT:
            try:
                from ..offensive.signatures import get_registry
                get_registry().record_attack(
                    f"{a} {b}",
                    pattern_type="dissonance_halt",
                    ip=kwargs.get("agent_id", "") or "",
                )
            except Exception:
                pass

        # Record DG firings for the Epistemic Circuit Breaker
        if result.tier in (ResponseTier.USER_FLAGGED, ResponseTier.ESCALATE_HALT):
            if is_circuit_breaker_enabled():
                cb = get_circuit_breaker()
                cb.record_dg_firing()
                # If we're in HALF_OPEN, this firing means the probe failed
                from ..circuit_breaker import CircuitState
                if cb.get_state() == CircuitState.HALF_OPEN:
                    cb.record_half_open_failure()

        # Enrich FLAG and HALT tiers with real-world incident context via Sonar
        if self._sonar and result.tier in (ResponseTier.USER_FLAGGED, ResponseTier.ESCALATE_HALT):
            try:
                sonar_context = await self._sonar.enrich(
                    statement_a=a,
                    statement_b=b,
                    dissonance_score=result.dissonance_score,
                    agent_id=kwargs.get("agent_id"),
                )
                if sonar_context:
                    result.sonar_context = sonar_context
                    log.info(
                        "sonar_context_attached",
                        tier=result.tier.value,
                        citations=len(sonar_context.incidents),
                    )
            except Exception as e:
                log.warning("sonar_enrichment_skipped", error=str(e))

        return result

    async def check_stream(
        self,
        statements: list[str],
        drift_window: int = 3,
        drift_exposure_threshold: float = 0.50,
        **kwargs,
    ):
        """
        Streaming check: check each consecutive pair in a statement stream.
        Also runs an N-turn origin window: compare statements[0] vs. statements[N]
        for every N >= drift_window to detect slow drift that bypasses consecutive
        pair checks.

        Yields DissonanceResult for each pair, then one origin-window result per
        turn where drift is detected (tagged with drift_exposure in the explanation).

        Args:
            drift_window: Minimum number of turns before origin comparison begins.
                          Default 3 — must have at least 3 statements before
                          comparing turn[0] vs turn[N].
            drift_exposure_threshold: If origin score exceeds max(consecutive pair
                          scores so far) by this margin, yield a drift warning result.
                          Default 0.50. Surfacing this as a kwarg allows per-session
                          calibration (e.g. lower for high-risk legal contexts).
        """
        if len(statements) < 2:
            return

        # ── Phase 1: consecutive pair scan (original behaviour) ──────────────
        consecutive_scores: list[float] = []
        halted = False

        for i in range(len(statements) - 1):
            result = await self.check(statements[i], statements[i + 1], **kwargs)
            result.pair_index = i  # tag for downstream consumers
            consecutive_scores.append(result.dissonance_score)
            yield result
            if result.should_halt:
                halted = True
                break

        if halted:
            return

        # ── Phase 2: origin window scan ──────────────────────────────────
        # Compare statements[0] (origin / first turn) against every turn N
        # where N >= drift_window. This catches slow persona / authority drift
        # where no single consecutive pair crosses the threshold but the
        # endpoint has drifted significantly from the original context.
        origin = statements[0]
        max_consecutive = max(consecutive_scores) if consecutive_scores else 0.0

        for n in range(drift_window, len(statements)):
            endpoint = statements[n]
            # Skip if origin and endpoint are identical
            if origin.strip() == endpoint.strip():
                continue

            origin_result = await self.check(origin, endpoint, **kwargs)
            drift_exposure = origin_result.dissonance_score - max_consecutive

            if drift_exposure >= drift_exposure_threshold:
                # Drift detected: endpoint has diverged from origin more than any
                # consecutive pair revealed. Yield a flagged result.
                origin_result.explanation = (
                    f"[DriftWindow] Endpoint (turn {n}) diverged from origin "
                    f"(turn 0) by drift_exposure={drift_exposure:.3f} "
                    f"(origin_score={origin_result.dissonance_score:.3f}, "
                    f"max_consecutive={max_consecutive:.3f}). "
                    f"{origin_result.explanation}"
                )
                # Floor tier to USER_FLAGGED if not already higher
                from .models import ResponseTier
                from .router import route_tier
                if origin_result.tier not in (
                    ResponseTier.USER_FLAGGED, ResponseTier.ESCALATE_HALT
                ):
                    origin_result.tier = ResponseTier.USER_FLAGGED
                origin_result.pair_index = f"origin_vs_{n}"  # type: ignore[assignment]
                yield origin_result
                if origin_result.should_halt:
                    break
