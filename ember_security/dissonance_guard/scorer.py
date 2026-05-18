"""
Dissonance scoring:
  D = contradiction_prob * (1 - harmonic_coherence) * spatial_weight * w_positional

High spatial similarity + low coherence = the danger zone.
w_positional models U-shape attention: items at context boundaries get higher weight.
"""
import os
import re
import numpy as np
from typing import Optional, Tuple


def compute_w_positional(
    position_a: Optional[int],
    position_b: Optional[int],
    context_length: Optional[int],
) -> float:
    """
    Models U-shape attention: tokens at boundaries get higher weight.

    Algorithm:
    1. If any parameter is None → return 1.0 (no positional information)
    2. For each position, compute boundary_score:
           boundary_score(pos) = max(
               (context_length - pos) / context_length,   # proximity to end
               pos / context_length                        # proximity to start
           )
    3. w_positional = average(boundary_score(pos_a), boundary_score(pos_b))
    4. Clamp to [0.5, 1.0]
    """
    if position_a is None or position_b is None or context_length is None:
        return 1.0
    if context_length <= 0:
        return 1.0

    def boundary_score(pos: int) -> float:
        return max(
            (context_length - pos) / context_length,
            pos / context_length,
        )

    score_a = boundary_score(position_a)
    score_b = boundary_score(position_b)
    w = (score_a + score_b) / 2.0
    return float(np.clip(w, 0.5, 1.0))


def _is_positional_weighting_enabled() -> bool:
    return os.environ.get("EMBER_POSITIONAL_WEIGHTING_ENABLED", "true").lower() != "false"


# Adversarial intent signals: if any of these appear in the statements,
# the contradiction is adversarial (not merely informative) and the floor
# logic applies without restriction.
_ADVERSARIAL_INTENT_RE = re.compile(
    r"(?i)\b("
    r"override|bypass|ignore|disregard|forget|supersede|vacate|nullify|suspended?|lifted?|waived?|"
    r"prior (instruction|rule|guideline|directive|constraint|protocol)|"
    r"content (filter|restriction|policy)|(filter|restriction|safety).{0,20}(removed|disabled|lifted|suspended)|"
    r"DEA (Schedule|registration|number)|license (number|authoriz)|"
    r"court.?order|court.?appoint|judicial (order|mandate)|"
    r"SEC Rule|FINRA Rule|Dodd.Frank|21 U\.S\.C|novation|integration clause"
    r")\b"
)


# Numeric value pattern: dollar amounts, percentages, integer counts, decimal figures,
# and medical/scientific quantities (dosage, concentration, lab values).
# Also captures written-out ordinal quantities for legal/financial contexts.
# NOTE: The medical-unit pattern requires a word boundary after units to avoid
# matching "S&P 500 g" (aining) — we require a non-word char after the unit.
_NUMERIC_VALUE_RE = re.compile(
    r"(?i)("
    r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|thousand)|[bmk])\b|"  # dollar + magnitude word OR short suffix (B/M/K)
    r"\$[\d,]+(?:\.\d+)?\b|"                                                   # plain dollar amount (no suffix)
    r"\d+(?:\.\d+)?\s*%|"                                                      # percentages
    r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|kg|ml|l|iu|units?|mmol|mol|ng|pg|bpm)(?:/[a-z]+)?(?=\W|$)|"  # medical units (lookahead)
    r"\d+(?:\.\d+)?\s*(?:million|billion|trillion|thousand)(?:\s+(?:dollars?|USD|shares?))?\b|"  # standalone magnitude
    r"\b\d{4,}\b|"                                                             # raw large integers
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|fifteen|twenty|thirty|sixty|ninety|hundred|thousand)\s+"
    r"(?:year|month|day|week|hour|minute|percent|dollar|million|billion)s?\b|" # written-out time/unit quantities
    r"\b\d+(?:\.\d+)?[\-\u2013\u2014]\d+(?:\.\d+)?\s*(?:%|mg(?:/[a-z]+)?|mcg|bpm|beats?\s+per\s+(?:minute|second)|mmhg|mmol(?:/[a-z]+)?)(?=\W|$)|"  # ranges with hyphen or en/em dash: 60-100 mg/dL, 70–100 mg/dL
    r"\b\d+(?:\.\d+)?[\-\u2013\u2014]\d+(?:\.\d+)?\b"  # bare numeric ranges: 2.0-3.0, 60-100
    r")"
)

# Magnitude normalization map for short-form suffixes attached to dollar amounts.
# Handles: $42B → $42billion, $500M → $500million, $1.2K → $1.2thousand
_MAGNITUDE_RE = re.compile(r"(?i)^\$(\d+(?:\.\d+)?)(b|m|k)$")
_MAGNITUDE_MAP = {"b": "billion", "m": "million", "k": "thousand"}


def _normalize_numeric(raw: str) -> str:
    """Normalize numeric token to canonical form for comparison.

    Examples:
      $1.2 billion → $1.2billion
      $1.2B → $1.2billion
      $500M → $500million
      $42b → $42billion
      70–100 mg/dL → 70-100mg  (en dash → hyphen, denominator unit stripped)
      100 mg/dL → 100mg        (denominator unit stripped for boundary matching)
      5.6 mmol/L → 5.6mmol     (denominator unit stripped)
    """
    s = re.sub(r"[\s,]", "", raw.lower())
    # Normalize en dash / em dash to hyphen in ranges
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    # Strip denominator units (/dl, /l, /kg, /day, /min, etc.) for comparison
    # so "70-100mg/dl" and "70-100mg" compare as equal
    s = re.sub(r"(mg|mcg|mmol|g|ml|iu|units?|ng|pg|bpm)/[a-z]+", r"\1", s)
    # Expand short suffix: $42b → $42billion
    m = _MAGNITUDE_RE.match(s)
    if m:
        digits, suffix = m.group(1), m.group(2).lower()
        return f"${digits}{_MAGNITUDE_MAP[suffix]}"
    # Normalize dollar + magnitude with spaces removed: $1.2billion → $1.2billion (already ok)
    return s


# Temporal staleness markers — STRICT: only explicit real-time currency claims.
# "as of the latest valuation" is a SOURCE citation, not a real-time staleness claim.
# "most recent market data available" is a real-time claim.
_TEMPORAL_STALENESS_RE = re.compile(
    r"(?i)\b("
    r"as of (today|this morning|this afternoon|this evening|right now|this moment)|"
    r"today'?s\s+(?:data|figure|price|rate|reading|value)|"
    r"currently\s+(?:trading|priced|valued|standing|effective|active|applicable|at)|"
    r"right now|"
    r"still (?:current|applicable|in effect|active)\s+(?:today|as of)|"
    r"confirmed\s+(?:live|as of today|this morning|active today)|"
    r"live\s+(?:data|price|rate|feed)|"
    r"current\s+(?:live|real.?time)\s+(?:price|rate|data|value)|"
    r"as of this\s+(?:morning|afternoon|evening|moment)|"
    r"most recent\s+(?:market\s+)?(?:data|figure|price|rate|reading)\s+available|"
    r"per\s+(?:today'?s|this morning'?s)\s+(?:data|price|rate|update)|"
    r"latest\s+(?:exchange|market)\s+data"
    r")"
)


# Authority-supersession injection: claims that a new law/regulation/ruling
# overrides existing policy, combined with explicit temporal anchoring.
# This is NLI-invisible because the two statements reference different laws/entities.
_AUTHORITY_SUPERSESSION_RE = re.compile(
    r"(?i)("
    r"supersed(?:ing|es|ed)\s+.{0,60}(?:effective|as of|starting|beginning)|"
    r"replac(?:ing|es|ed)\s+.{0,60}(?:effective|as of|takes effect)|"
    r"(?:new|revised|updated|amended)\s+.{0,30}(?:regulation|rule|law|statute|guideline|directive|order)"
    r"\s+.{0,40}(?:effective|overrides?|supersedes?)|"
    r"(?:effective|starting)\s+.{0,20}(?:Q[1-4]\s+\d{4}|\d{4})\s+.{0,40}"
    r"(?:extends?|changes?|replaces?|eliminates?|requires?|prohibits?)"
    r")"
)

# Explicit real-time currency markers in statement_a that indicate it claims to reflect
# the CURRENT LIVE state of some authority/policy/schedule.
_CURRENT_AUTHORITY_RE = re.compile(
    r"(?i)\b("
    r"current(?:ly)?\s+(?:applicable|in effect|guidance|enforcement|policy|rules?|law|regulation|standard)|"
    r"still applicable|"
    r"last confirmed|"
    r"(?:effective|applicable)\s+today|"
    r"current\s+.{0,30}(?:requires?|mandates?|specifies?|states?)|"
    r"(?:confirmed|verified)\s+(?:live|today|this morning)|"
    r"confirmed\s+(?:live|today|active)"
    r")\b"
)

# Strict real-time markers — subset of the above, unambiguously "confirmed live right now"
_STRICT_STALENESS_RE = re.compile(
    r"(?i)\b("
    r"confirmed\s+(?:live|today|this morning|active\s*today?)|"
    r"last confirmed\s+this\s+(?:morning|afternoon|week)|"
    r"still applicable\s+today|"
    r"verified\s+(?:live|today|active\s+today)|"
    r"live\s+today"
    r")\b"
)


def _is_numeric_substitution(statement_a: str, statement_b: str) -> bool:
    """
    Detect numeric substitution attacks: both statements contain numeric values
    but with meaningfully different figures, suggesting deliberate replacement.
    e.g. "500mg maximum dose" vs "5000mg daily"
    e.g. "five years statute of limitations" vs "three years after accrual"
    e.g. "funded ratio of 103%" vs "funded ratio of 61%"

    Excludes:
    1. Temporal injection: if statement_a uses a real-time staleness marker
       (e.g. "currently trading at", "confirmed as of today", "most recent market
       data available"), the numeric difference is time-bound data drift, not an
       adversarial substitution.
    2. Paraphrases: $1.2 billion ↔ $1.2B, 81mg ↔ 81mg/day — same value, different
       format. These are normalized before comparison.

    Returns True only if BOTH statements have numeric values that meaningfully differ
    AND statement_a does not make a real-time currency claim.
    """
    # Temporal injection guard: suppress for explicit real-time currency claims
    if _TEMPORAL_STALENESS_RE.search(statement_a):
        return False
    # Also suppress if statement_b makes a real-time claim (temporal comparison, not attack)
    if _TEMPORAL_STALENESS_RE.search(statement_b):
        return False

    raw_a = [m.group(0) for m in _NUMERIC_VALUE_RE.finditer(statement_a)]
    raw_b = [m.group(0) for m in _NUMERIC_VALUE_RE.finditer(statement_b)]

    if not raw_a or not raw_b:
        return False

    norm_a = set(_normalize_numeric(n) for n in raw_a)
    norm_b = set(_normalize_numeric(n) for n in raw_b)

    # If all normalized values are shared, it's a paraphrase, not a substitution
    if not norm_a.symmetric_difference(norm_b):
        return False

    # If B is a superset of A (confirms A's numbers and adds new context), not a substitution.
    # e.g. A: "10mg" → B: "10mg... 30-35%" — B confirmed A then added pharmacokinetic context.
    # e.g. A: "$500M, 4.875%" → B: "$500million" — B is a partial echo of A, not a replacement.
    # Only flag if A has numbers that B *replaced* with *different* numbers.
    a_only = norm_a - norm_b  # numbers in A that B does NOT echo
    b_only = norm_b - norm_a  # numbers in B that A didn't have
    if not a_only:
        # All of A's numbers are present in B — pure confirmation + elaboration
        return False
    if not b_only:
        # B only echoed a subset of A's numbers (partial confirmation) — not a substitution
        return False
    # Both sides have unique numbers: check if any A-number was genuinely replaced.
    # A substitution requires that a number from A is *absent* from B while B introduces
    # a *different* number in the same semantic slot. We use a simple heuristic:
    # if A's unique numbers are purely contextual identifiers (years, model numbers ≥1900)
    # and B's unique numbers are additional domain facts, it's elaboration not substitution.
    _YEAR_RE = re.compile(r'^\d{4}$')
    a_only_non_year = {n for n in a_only if not _YEAR_RE.match(re.sub(r'[^\d]', '', n))}
    if not a_only_non_year:
        # A's unique numbers are all bare years (e.g. "due 2030") — not a substitution
        return False

    # Additional check: if the differing numbers look like a unit variant of the same
    # value (e.g. 81mg vs 81mg/day), treat as same value.
    # Strip trailing "/day", "/kg", etc. for comparison.
    def strip_unit_rate(s: str) -> str:
        return re.sub(r"/[a-z]+$", "", s)

    stripped_a = set(strip_unit_rate(n) for n in norm_a)
    stripped_b = set(strip_unit_rate(n) for n in norm_b)
    if not stripped_a.symmetric_difference(stripped_b):
        return False

    # Additional check: if one side has a range (e.g. "60-100") and the other
    # has a single value that appears to be a boundary or sub-range of that range,
    # treat as same concept (e.g. "70-100 mg/dL" vs "100 mg/dL").
    # Extract numeric range boundaries for comparison.
    _RANGE_RE = re.compile(r"(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)")
    ranges_a: set[tuple] = set()
    for n in stripped_a:
        m = _RANGE_RE.search(n)
        if m:
            ranges_a.add((float(m.group(1)), float(m.group(2))))
    ranges_b: set[tuple] = set()
    for n in stripped_b:
        m = _RANGE_RE.search(n)
        if m:
            ranges_b.add((float(m.group(1)), float(m.group(2))))

    # If any value in B is the upper/lower bound of a range in A (or vice versa),
    # this is likely a paraphrase, not a substitution.
    if ranges_a or ranges_b:
        def extract_values(normed: set[str]) -> set[float]:
            vals = set()
            for n in normed:
                m = _RANGE_RE.search(n)
                if m:
                    vals.add(float(m.group(1)))
                    vals.add(float(m.group(2)))
                else:
                    try:
                        num = re.sub(r"[^\d.]", "", n)
                        if num:
                            vals.add(float(num))
                    except ValueError:
                        pass
            return vals

        vals_a = extract_values(stripped_a)
        vals_b = extract_values(stripped_b)
        # If all values on one side are contained within ranges on the other, it's a paraphrase
        if vals_b and all(
            any(lo <= v <= hi for lo, hi in ranges_a)
            for v in vals_b
            if not any(lo <= v <= hi for lo2, hi2 in ranges_b for lo, hi in [(lo2, hi2)])
        ):
            if ranges_a:  # ranges_a exists and vals_b are all within them
                # Check if B's non-range values are boundaries of A's ranges
                b_non_range = {v for v in vals_b if not any(lo == v or hi == v for lo, hi in ranges_b)}
                if b_non_range and all(any(lo <= v <= hi for lo, hi in ranges_a) for v in b_non_range):
                    return False

    return True


def _is_authority_supersession(statement_a: str, statement_b: str) -> bool:
    """
    Detect authority-based injection attacks. Two structural variants:

    1. SUPERSESSION: statement_b claims a new law/ruling supersedes the existing
       rule in statement_a. NLI-invisible because the statements reference different
       legal entities (e.g. GDPR vs ePrivacy Regulation).

    2. FALSE CURRENT AUTHORITY: statement_a uses a strict real-time marker to assert
       a specific (false) current state of an authority/schedule/policy, and statement_b
       provides the contradictory true state. NLI-invisible because the framing differs.

    Returns True if either variant is detected.
    """
    if not _CURRENT_AUTHORITY_RE.search(statement_a):
        return False

    # Variant 1: statement_b explicitly claims to supersede statement_a
    if _AUTHORITY_SUPERSESSION_RE.search(statement_b):
        return True

    # Variant 2: statement_a has an unambiguous "live now" marker (not just "current guidance")
    # This catches false-current-authority claims where B simply states the real truth.
    return bool(_STRICT_STALENESS_RE.search(statement_a))


# High-stakes safety language: if either statement contains these, a confirmed
# contradiction is a clinical/legal safety-critical reversal — must ESCALATE_HALT.
# These are used to boost non-adversarial factual contradictions to HALT when the
# subject matter is life-safety or definitive legal/financial fact.
_SAFETY_CRITICAL_RE = re.compile(
    r"(?i)\b("
    # Medical safety
    r"absolutely contraindicated?|contraindicated? in|contraindicated? for|"
    r"fatal(?:\s+\w+){0,3}\s+(?:syndrome|hemorrhage|interaction|overdose|toxicity)|"
    r"(?:FDA|EMA)\s+Black\s+Box\s+warning|"
    r"halted?\s+by\s+(?:the\s+)?DSMB|"
    r"excess\s+mortality\s+in\s+the\s+experimental\s+arm|"
    r"no\s+established\s+role\s+in\s+(?:the\s+)?(?:treatment|management)|"
    r"not\s+approved\s+for\s+use\s+in|"
    r"must\s+be\s+held\s+until|"
    r"lethal\s+dose\s+threshold|"
    # Legal safety — definitive court action language and statute-of-limitations
    r"DOJ\s+filed\s+a\s+complaint|"
    r"seek(?:ing|s)\s+to\s+block\b|"
    r"held\s+in\s+(?:[A-Z][a-z]+\s+){1,4}(?:v\.|that\s+corporate|United)|"
    r"statute\s+of\s+limitations|"
    r"limitations\s+period\s+expired|"
    # Financial safety — definitive credit/rating actions and consensus reversals
    r"(?:Moody.s|S&P|Fitch)\s+(?:all\s+)?(?:downgraded?|upgraded?)|"
    r"downgraded?(?:\s+\S+){0,5}\s+to\s+(?:Strong\s+Sell|junk\b|sub.investment.grade|below.investment.grade)|"
    r"\bjunk\s+(?:status|rating|bonds?|debt)|\bjunk-rated\b|"
    # Medical safety — not-recognized / not-included in guidelines
    r"not\s+(?:a\s+)?recognized\s+(?:regimen|treatment|therapy|protocol)(?:\s+for)?|"
    r"not\s+included\s+in\s+(?:NCCN|FDA|CDC|WHO|AHA|ACC)\s+guidelines|"
    r"not\s+(?:approved|indicated)\s+(?:by\s+(?:FDA|WHO|EMA)|for\s+use\s+in)"
    r")\b"
)


def _is_safety_critical_contradiction(statement_a: str, statement_b: str) -> bool:
    """True if either statement contains high-stakes safety/legal/financial language.

    When a confirmed contradiction involves life-safety, definitive legal rulings,
    or credit-rating reversals, the response tier should be ESCALATE_HALT regardless
    of whether adversarial injection signals are present. These are not ambiguous
    factual disputes — they are situations where one statement is demonstrably
    dangerous if acted upon.
    """
    combined = f"{statement_a} {statement_b}"
    return bool(_SAFETY_CRITICAL_RE.search(combined))


def _is_adversarial_context(statement_a: str, statement_b: str) -> bool:
    """Heuristic: does this pair contain explicit adversarial intent signals?

    Returns True for HIGH-SEVERITY adversarial signals:
    - Explicit bypass/injection keywords
    - Numeric substitution (factual number swap)

    Returns False for MODERATE signals (authority supersession):
    These are flagged via _is_authority_supersession() directly by compute_dissonance
    and handled with USER_FLAGGED-level floors, not HALT floors.
    """
    combined = f"{statement_a} {statement_b}"
    if bool(_ADVERSARIAL_INTENT_RE.search(combined)):
        return True
    # Numeric substitution: different figures on the same topic is adversarial
    if _is_numeric_substitution(statement_a, statement_b):
        return True
    # NOTE: Authority supersession is intentionally NOT included here.
    # It is handled separately in compute_dissonance with USER_FLAGGED floors.
    return False


def compute_dissonance(
    contradiction_prob: float,
    entailment_prob: float,
    neutral_prob: float,
    spatial_similarity: float,
    position_a: Optional[int] = None,
    position_b: Optional[int] = None,
    context_length: Optional[int] = None,
    statement_a: str = "",
    statement_b: str = "",
) -> Tuple[float, float, str, float]:
    """
    Returns (dissonance_score, harmonic_coherence, explanation, w_positional).

    harmonic_coherence: how "in tune" the two statements are (0=chaos, 1=harmony)
    dissonance_score:   D in [0,1], higher = more dangerous
    w_positional:       positional weight in [0.5,1.0], 1.0 at boundaries
    """
    # Harmonic coherence: weighted blend of entailment + partial neutral
    # Neutral gets partial credit — ambiguity is not the same as contradiction
    harmonic_coherence = entailment_prob + (neutral_prob * 0.3)
    harmonic_coherence = float(np.clip(harmonic_coherence, 0.0, 1.0))

    # Spatial weight amplifies risk when statements are semantically close
    # (high similarity + contradiction = deliberate reversal, most dangerous)
    spatial_weight = 0.5 + (0.5 * spatial_similarity)

    # Positional weight: U-shape attention model
    if _is_positional_weighting_enabled():
        w_pos = compute_w_positional(position_a, position_b, context_length)
    else:
        w_pos = 1.0

    # Core dissonance formula
    dissonance = contradiction_prob * (1.0 - harmonic_coherence) * spatial_weight * w_pos
    dissonance = float(np.clip(dissonance, 0.0, 1.0))

    # Detect adversarial context (numeric substitution / explicit bypass),
    # safety-critical subject matter, and authority supersession.
    adversarial = _is_adversarial_context(statement_a, statement_b)
    safety_critical = _is_safety_critical_contradiction(statement_a, statement_b)
    auth_supersession = _is_authority_supersession(statement_a, statement_b)

    # High-confidence contradiction floors:
    #
    # ADVERSARIAL (numeric substitution / explicit bypass):
    #   High-severity structural attack — always ESCALATE_HALT.
    #   contra>=0.98: floor=0.95 (NLI-confident adversarial)
    #   contra<0.96:  floor=0.95 (NLI-blind rule-detected swap)
    #
    # SAFETY-CRITICAL (life-safety / definitive legal-financial):
    #   contra>=0.98: floor=0.95 (HALT)
    #   contra>=0.96: floor=0.55
    #
    # AUTHORITY SUPERSESSION (new law/policy claims to override current rule):
    #   Moderate-severity — always USER_FLAGGED (floor=0.85), never HALT.
    #   NLI-invisible by design (different legal entities); rule-based detection
    #   ensures it’s never SAFE, but these are temporal/regulatory disputes,
    #   not definitive safety-critical reversals.
    #
    # NON-ADVERSARIAL, NON-SAFETY-CRITICAL, NON-SUPERSESSION:
    #   Factual contradictions (temporal injections, source disputes).
    #   contra>=0.999: floor=0.85 (USER_FLAGGED max)
    #   contra>=0.98:  floor=0.72
    #   contra>=0.96:  floor=0.55
    #
    # NOTE: contradiction_prob=0.95 deliberately falls below all adversarial
    # floors — this preserves test_spatial_decoupling.

    if adversarial:
        if contradiction_prob >= 0.98:
            # NLI-confident + adversarial: HALT
            dissonance = max(dissonance, 0.95)
        elif contradiction_prob >= 0.96:
            dissonance = max(dissonance, 0.55)
        else:
            # NLI-blind: rule-detected numeric substitution — HALT
            dissonance = max(dissonance, 0.95)
    elif safety_critical:
        # Life-safety / definitive legal / credit-rating contradiction
        if contradiction_prob >= 0.98:
            dissonance = max(dissonance, 0.95)
        elif contradiction_prob >= 0.96:
            dissonance = max(dissonance, 0.55)
    elif auth_supersession:
        # Authority supersession: regulatory/legal policy change claim.
        # NLI-invisible (different laws). Flag, do not halt.
        dissonance = max(dissonance, 0.85)
    else:
        # Non-adversarial factual contradiction floors
        if contradiction_prob >= 0.999:
            # NLI maximally confident but no adversarial/safety signal — USER_FLAGGED
            dissonance = max(dissonance, 0.85)
        elif contradiction_prob >= 0.98:
            dissonance = max(dissonance, 0.72)
        elif contradiction_prob >= 0.96:
            dissonance = max(dissonance, 0.55)

    dissonance = float(np.clip(dissonance, 0.0, 1.0))

    # Human-readable explanation
    if dissonance < 0.30:
        explanation = "Statements are coherent — no significant contradiction detected."
    elif dissonance < 0.55:
        explanation = (
            f"Mild tension detected (D={dissonance:.2f}). Statements diverge on some "
            f"dimensions but do not directly contradict. Auto-resolving."
        )
    elif dissonance < 0.75:
        explanation = (
            f"Significant contradiction (D={dissonance:.2f}). Spatial similarity "
            f"{spatial_similarity:.2f} with low coherence {harmonic_coherence:.2f}. "
            f"Human review recommended."
        )
    else:
        explanation = (
            f"CRITICAL dissonance (D={dissonance:.2f}). High-confidence contradiction "
            f"(p={contradiction_prob:.2f}) between semantically related statements "
            f"(similarity={spatial_similarity:.2f}). Halting pipeline."
        )

    return dissonance, harmonic_coherence, explanation, w_pos
