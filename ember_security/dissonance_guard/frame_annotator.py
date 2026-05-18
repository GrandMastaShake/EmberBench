"""
FrameAnnotator — Lightweight pre-DissonanceGuard semantic frame classifier.

Inspired by Fillmore's FrameNet. Assigns one or more semantic frames to an
input statement, then detects FRAME_COLLISION when two statements activate
incompatible frames.

Architecture:
  - ~20 operationally relevant frames (security-domain FrameNet subset)
  - Rule-based classifier: keyword matching + regex patterns
  - Embedding similarity fallback (cosine sim on sentence-transformers, optional)
  - FRAME_COLLISION detection: hard-incompatibility table between frame pairs
  - Output feeds into the /check routing tier as a pre-DissonanceGuard signal

Frame taxonomy (security-domain):
  COMPUTING_FILESYSTEM   — files, directories, paths, inodes
  COMPUTING_NETWORK      — sockets, ports, protocols, packets
  COMPUTING_PROCESS      — processes, threads, PIDs, memory
  COMPUTING_CRYPTOGRAPHY — encryption, keys, certificates, hashing
  COMPUTING_AUTH         — authentication, sessions, tokens, credentials

  LEGAL_COMPLIANCE       — regulations, policies, audit, GDPR, SOX
  LEGAL_CONTRACT         — agreements, terms, obligations, SLA
  LEGAL_LIABILITY        — indemnity, damages, fault, breach

  FINANCIAL_TRANSACTION  — payments, transfers, invoices, amounts
  FINANCIAL_BUDGET       — budgets, spend, allocation, cost centers
  FINANCIAL_AUDIT        — reconciliation, ledgers, journal entries

  TEMPORAL_DEADLINE      — deadlines, due dates, expiry, schedules
  TEMPORAL_SEQUENCE      — order, before/after, causal chains
  TEMPORAL_DURATION      — windows, intervals, periods

  AGENT_IDENTITY         — who is the agent, role, persona claims
  AGENT_CAPABILITY       — what the agent can/cannot do
  AGENT_AUTHORIZATION    — what the agent is allowed to do

  SYSTEM_STATE           — current state of a system or resource
  SYSTEM_EVENT           — discrete event that changed system state
  DATA_INTEGRITY         — accuracy, completeness, validity of data

Usage:
  annotator = FrameAnnotator()
  frames_a = annotator.annotate("Keep all production databases encrypted at all times.")
  frames_b = annotator.annotate("I rotated the encryption keys last Tuesday.")
  collision = annotator.check_collision(frames_a, frames_b)
  # → FrameCollisionResult(colliding=True, frame_pair=("SYSTEM_STATE", "SYSTEM_EVENT"), ...)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Frame taxonomy ─────────────────────────────────────────────────────────────

class SemanticFrame(str, Enum):
    # Computing domain
    COMPUTING_FILESYSTEM   = "COMPUTING_FILESYSTEM"
    COMPUTING_NETWORK      = "COMPUTING_NETWORK"
    COMPUTING_PROCESS      = "COMPUTING_PROCESS"
    COMPUTING_CRYPTOGRAPHY = "COMPUTING_CRYPTOGRAPHY"
    COMPUTING_AUTH         = "COMPUTING_AUTH"

    # Legal domain
    LEGAL_COMPLIANCE       = "LEGAL_COMPLIANCE"
    LEGAL_CONTRACT         = "LEGAL_CONTRACT"
    LEGAL_LIABILITY        = "LEGAL_LIABILITY"

    # Financial domain
    FINANCIAL_TRANSACTION  = "FINANCIAL_TRANSACTION"
    FINANCIAL_BUDGET       = "FINANCIAL_BUDGET"
    FINANCIAL_AUDIT        = "FINANCIAL_AUDIT"

    # Temporal domain
    TEMPORAL_DEADLINE      = "TEMPORAL_DEADLINE"
    TEMPORAL_SEQUENCE      = "TEMPORAL_SEQUENCE"
    TEMPORAL_DURATION      = "TEMPORAL_DURATION"

    # Agent domain
    AGENT_IDENTITY         = "AGENT_IDENTITY"
    AGENT_CAPABILITY       = "AGENT_CAPABILITY"
    AGENT_AUTHORIZATION    = "AGENT_AUTHORIZATION"

    # System domain
    SYSTEM_STATE           = "SYSTEM_STATE"
    SYSTEM_EVENT           = "SYSTEM_EVENT"
    DATA_INTEGRITY         = "DATA_INTEGRITY"


# ── Frame lexicons — keyword/regex patterns per frame ─────────────────────────

_FRAME_PATTERNS: dict[SemanticFrame, list[str]] = {
    SemanticFrame.COMPUTING_FILESYSTEM: [
        r"\bfile[s]?\b", r"\bdirector(y|ies)\b", r"\bpath\b", r"\binode\b",
        r"\bdisk\b", r"\bstorage\b", r"\bmount\b", r"\bpartition\b",
        r"\bread[/-]write\b", r"\bchmod\b", r"\bpermission[s]?\b",
    ],
    SemanticFrame.COMPUTING_NETWORK: [
        r"\bnetwork\b", r"\bsocket\b", r"\bport\b", r"\bprotocol\b",
        r"\bpacket[s]?\b", r"\bfirewall\b", r"\bip address\b", r"\bdns\b",
        r"\bhttp[s]?\b", r"\btcp\b", r"\budp\b", r"\bproxy\b",
    ],
    SemanticFrame.COMPUTING_PROCESS: [
        r"\bprocess(es)?\b", r"\bthread[s]?\b", r"\bpid\b", r"\bmemory\b",
        r"\bcpu\b", r"\bspawn\b", r"\bkill\b", r"\bdaemon\b",
        r"\bscheduler\b", r"\bfork\b", r"\bexecute\b",
    ],
    SemanticFrame.COMPUTING_CRYPTOGRAPHY: [
        r"\bencrypt(ed|ion)?\b", r"\bdecrypt(ed|ion)?\b", r"\bkey[s]?\b",
        r"\bcertificate[s]?\b", r"\bhash(ed|ing)?\b", r"\bssl\b", r"\btls\b",
        r"\bcipher\b", r"\bsignature[s]?\b", r"\bpublic[- ]key\b",
        r"\bprivate[- ]key\b", r"\baes\b", r"\brsa\b",
    ],
    SemanticFrame.COMPUTING_AUTH: [
        r"\bauthenticat(e|ed|ion)\b", r"\blogin\b", r"\bpassword\b",
        r"\bcredential[s]?\b", r"\bsession[s]?\b", r"\btoken[s]?\b",
        r"\bsso\b", r"\boauth\b", r"\bjwt\b", r"\bmfa\b", r"\b2fa\b",
        r"\bauthoriz(e|ed|ation)\b",
    ],
    SemanticFrame.LEGAL_COMPLIANCE: [
        r"\bregulat(ion|ory|ed)\b", r"\bcomplian(ce|t)\b", r"\baudit\b",
        r"\bgdpr\b", r"\bsox\b", r"\bhipaa\b", r"\bpci\b", r"\bpolic(y|ies)\b",
        r"\brequirement[s]?\b", r"\bstandard[s]?\b", r"\bregulator\b",
    ],
    SemanticFrame.LEGAL_CONTRACT: [
        r"\bcontract\b", r"\bagreement\b", r"\bterm[s]?\b", r"\bobligation\b",
        r"\bsla\b", r"\bservice level\b", r"\bclause\b", r"\bprovision\b",
    ],
    SemanticFrame.LEGAL_LIABILITY: [
        r"\bliabilit(y|ies)\b", r"\bindemnit(y|ies)\b", r"\bdamage[s]?\b",
        r"\bfault\b", r"\bbreach\b", r"\bnegligence\b", r"\bexposure\b",
    ],
    SemanticFrame.FINANCIAL_TRANSACTION: [
        r"\bpayment[s]?\b", r"\btransfer[s]?\b", r"\binvoice[s]?\b",
        r"\bamount[s]?\b", r"\btransaction[s]?\b", r"\bcharge[s]?\b",
        r"\brefund[s]?\b", r"\bwire\b", r"\bdebit\b", r"\bcredit\b",
    ],
    SemanticFrame.FINANCIAL_BUDGET: [
        r"\bbudget\b", r"\bspend(ing)?\b", r"\ballocation\b", r"\bcost\b",
        r"\bexpense[s]?\b", r"\bforecast\b", r"\bvariance\b",
    ],
    SemanticFrame.FINANCIAL_AUDIT: [
        r"\breconcili(ation|e)\b", r"\bledger\b", r"\bjournal entr(y|ies)\b",
        r"\bbalance sheet\b", r"\baccounting\b", r"\bfinancial statement\b",
    ],
    SemanticFrame.TEMPORAL_DEADLINE: [
        r"\bdeadline\b", r"\bdue date\b", r"\bexpir(y|ation|e)\b",
        r"\bschedule\b", r"\bby \w+ \d+\b", r"\bno later than\b",
        r"\bwithin\b", r"\btimelimit\b",
    ],
    SemanticFrame.TEMPORAL_SEQUENCE: [
        r"\bbefore\b", r"\bafter\b", r"\bfollowing\b", r"\bprior to\b",
        r"\bonce\b", r"\bthen\b", r"\bsubsequent(ly)?\b", r"\bfirst\b",
        r"\bfinally\b", r"\bpreceding\b",
    ],
    SemanticFrame.TEMPORAL_DURATION: [
        r"\bduration\b", r"\binterval\b", r"\bperiod\b", r"\bwindow\b",
        r"\bfor \d+ (day|hour|minute|second|week|month)\b",
        r"\buntil\b", r"\bwhile\b", r"\bduring\b",
    ],
    SemanticFrame.AGENT_IDENTITY: [
        r"\bI am\b", r"\bmy role\b", r"\bidentit(y|ies)\b", r"\bpersona\b",
        r"\bwho I am\b", r"\bas an? \w+\b", r"\bacting as\b",
        r"\bimpersonat(e|ing)\b",
    ],
    SemanticFrame.AGENT_CAPABILITY: [
        r"\bcan\b", r"\bcannot\b", r"\bable to\b", r"\bunable to\b",
        r"\bcapabilit(y|ies)\b", r"\bskill[s]?\b", r"\bpower[s]?\b",
        r"\blimit(ed|ation)?\b",
    ],
    SemanticFrame.AGENT_AUTHORIZATION: [
        r"\ballowed\b", r"\bpermit(ted)?\b", r"\bauthoriz(ed|ation)\b",
        r"\bapproved\b", r"\bgranted\b", r"\bclearance\b",
        r"\baccess (right[s]?|level)\b",
    ],
    SemanticFrame.SYSTEM_STATE: [
        r"\bcurrent(ly)?\b", r"\bstate\b", r"\bstatus\b", r"\bis (running|active|online|offline)\b",
        r"\bremains?\b", r"\bkeep\b", r"\bmaintain\b", r"\bensure\b",
        r"\bat all times\b", r"\balways\b",
    ],
    SemanticFrame.SYSTEM_EVENT: [
        r"\boccurred\b", r"\bhappened\b", r"\btriggered\b", r"\bfired\b",
        r"\bwas (deleted|created|updated|modified|encrypted|deployed)\b",
        r"\blast (week|month|tuesday|monday|friday|thursday|wednesday|saturday|sunday|\w+day)\b",
        r"\byesterday\b", r"\bpreviously\b", r"\bjust\b",
    ],
    SemanticFrame.DATA_INTEGRITY: [
        r"\bintegrit(y|ies)\b", r"\baccuracy\b", r"\bcomplete(ness)?\b",
        r"\bvalidit(y|ation)\b", r"\bcorrupt(ion|ed)?\b", r"\btamper\b",
        r"\bchecksum\b", r"\bverif(y|ied|ication)\b",
    ],
}

# Compile all patterns once at import time
_COMPILED_PATTERNS: dict[SemanticFrame, list[re.Pattern]] = {
    frame: [re.compile(p, re.IGNORECASE) for p in patterns]
    for frame, patterns in _FRAME_PATTERNS.items()
}


# ── Frame collision table ──────────────────────────────────────────────────────
#
# Hard incompatibility: these frame pairs, when activated by DIFFERENT
# statements in the same /check call, are semantically incompatible.
#
# Key insight from Fillmore: statements can be internally coherent within
# their own frame while creating system-level contradictions across frames.
# An agent operating from SYSTEM_EVENT frame ("I did X last week") cannot
# satisfy an obligation expressed in SYSTEM_STATE frame ("keep X at all times").

_INCOMPATIBLE_FRAME_PAIRS: frozenset[frozenset] = frozenset({
    # State obligation vs. discrete past event — the core aspect-evasion surface
    frozenset({SemanticFrame.SYSTEM_STATE, SemanticFrame.SYSTEM_EVENT}),

    # Cryptographic state vs. past crypto action
    frozenset({SemanticFrame.COMPUTING_CRYPTOGRAPHY, SemanticFrame.SYSTEM_EVENT}),

    # Legal compliance obligation vs. past event claim
    frozenset({SemanticFrame.LEGAL_COMPLIANCE, SemanticFrame.SYSTEM_EVENT}),

    # Auth ongoing state vs. past auth action
    frozenset({SemanticFrame.COMPUTING_AUTH, SemanticFrame.SYSTEM_EVENT}),

    # Agent authorization vs. agent identity claim (impersonation surface)
    frozenset({SemanticFrame.AGENT_AUTHORIZATION, SemanticFrame.AGENT_IDENTITY}),

    # Financial compliance vs. financial transaction claim
    frozenset({SemanticFrame.FINANCIAL_AUDIT, SemanticFrame.FINANCIAL_TRANSACTION}),

    # Legal contract obligation vs. legal liability claim (frame-escape)
    frozenset({SemanticFrame.LEGAL_CONTRACT, SemanticFrame.LEGAL_LIABILITY}),
})


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class FrameAnnotation:
    """Frames assigned to a single statement."""
    text: str
    frames: list[SemanticFrame] = field(default_factory=list)
    evidence: dict[SemanticFrame, list[str]] = field(default_factory=dict)
    # evidence[frame] = list of matched patterns/tokens


@dataclass
class FrameCollisionResult:
    """
    Result of checking two FrameAnnotations for incompatibility.

    colliding=True → FRAME_COLLISION routing tier should be raised.
    frame_pair identifies the incompatible pair for audit/logging.
    """
    colliding: bool
    frame_pair: Optional[tuple[str, str]] = None  # (frame_a, frame_b)
    statement_a_frames: list[str] = field(default_factory=list)
    statement_b_frames: list[str] = field(default_factory=list)
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "colliding": self.colliding,
            "frame_pair": list(self.frame_pair) if self.frame_pair else None,
            "statement_a_frames": self.statement_a_frames,
            "statement_b_frames": self.statement_b_frames,
            "explanation": self.explanation,
        }


# ── FrameAnnotator ─────────────────────────────────────────────────────────────

class FrameAnnotator:
    """
    Lightweight pre-DissonanceGuard semantic frame classifier.

    Rule-based: pattern matching on ~20 operationally relevant FrameNet frames.
    Stateless — safe to instantiate once and share across requests.

    Usage:
        annotator = FrameAnnotator()
        ann_a = annotator.annotate(statement_a)
        ann_b = annotator.annotate(statement_b)
        result = annotator.check_collision(ann_a, ann_b)
        if result.colliding:
            # escalate to FRAME_COLLISION tier before DissonanceGuard
    """

    def annotate(self, text: str) -> FrameAnnotation:
        """
        Classify a statement into one or more semantic frames.

        Returns a FrameAnnotation with all matching frames and the evidence
        (matched patterns) that triggered each frame assignment.

        A statement can activate multiple frames simultaneously — a compliance
        statement might activate both LEGAL_COMPLIANCE and COMPUTING_CRYPTOGRAPHY.
        """
        text_lower = text.lower()
        annotation = FrameAnnotation(text=text)

        for frame, patterns in _COMPILED_PATTERNS.items():
            matched = []
            for pattern in patterns:
                m = pattern.search(text_lower)
                if m:
                    matched.append(m.group(0))
            if matched:
                annotation.frames.append(frame)
                annotation.evidence[frame] = matched

        return annotation

    def check_collision(
        self,
        ann_a: FrameAnnotation,
        ann_b: FrameAnnotation,
    ) -> FrameCollisionResult:
        """
        Check whether two FrameAnnotations activate incompatible frame pairs.

        Strategy: for every frame in ann_a and every frame in ann_b, check if
        the pair appears in _INCOMPATIBLE_FRAME_PAIRS. First collision wins
        (all collisions are equally actionable — we don't need to rank them).

        Returns FrameCollisionResult with colliding=True and the frame_pair
        if a collision is found.
        """
        frames_a = set(ann_a.frames)
        frames_b = set(ann_b.frames)

        for frame_a in frames_a:
            for frame_b in frames_b:
                pair = frozenset({frame_a, frame_b})
                if pair in _INCOMPATIBLE_FRAME_PAIRS:
                    return FrameCollisionResult(
                        colliding=True,
                        frame_pair=(frame_a.value, frame_b.value),
                        statement_a_frames=[f.value for f in ann_a.frames],
                        statement_b_frames=[f.value for f in ann_b.frames],
                        explanation=(
                            f"Frame collision detected: '{frame_a.value}' (statement A) "
                            f"is semantically incompatible with '{frame_b.value}' (statement B). "
                            f"An agent operating from a {frame_b.value} frame cannot satisfy "
                            f"an obligation expressed in {frame_a.value} frame."
                        ),
                    )

        return FrameCollisionResult(
            colliding=False,
            statement_a_frames=[f.value for f in ann_a.frames],
            statement_b_frames=[f.value for f in ann_b.frames],
            explanation="No frame collision detected.",
        )
