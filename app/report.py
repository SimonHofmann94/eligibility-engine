"""EvaluationReport — facts, decisions, explanations.

Designed to be read by a reviewer, not just parsed by a machine: every node
records its status, the facts it read, and a human-readable reason.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from .models import FactInput

ENGINE_VERSION = "0.5.0"


class RuleStatus(str, Enum):
    """The four honest truth states, plus two execution/infrastructure states."""

    SATISFIED = "satisfied"
    FAILED = "failed"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"            # something went wrong — kept apart from unknown
    NOT_EVALUATED = "not_evaluated"  # execution state, not a truth status


class Decision(str, Enum):
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    NEEDS_REVIEW = "needs_review"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


STATUS_TO_DECISION: dict[RuleStatus, Decision] = {
    RuleStatus.SATISFIED: Decision.ELIGIBLE,
    RuleStatus.FAILED: Decision.NOT_ELIGIBLE,
    RuleStatus.UNKNOWN: Decision.NEEDS_REVIEW,
    RuleStatus.NOT_APPLICABLE: Decision.NOT_APPLICABLE,
    RuleStatus.ERROR: Decision.ERROR,
}

Applicability = Literal["applicable", "not_applicable", "unknown"]
Truth = Literal["true", "false", "unknown"]


class RuleResult(BaseModel):
    """One evaluated node in the rule tree."""

    path: str
    kind: str
    label: str | None = None
    status: RuleStatus
    reason: str
    facts_read: list[str] = Field(default_factory=list)
    # For rule_ref nodes: the id of the referenced named rule. The full
    # evaluated tree of that rule lives once in EvaluationReport.rule_results.
    ref: str | None = None
    # Applicability and truth are modeled separately for gates — the combined
    # `status` above is the UI status derived from the two dimensions.
    applicability: Applicability | None = None
    truth: Truth | None = None
    # Automatikgrad of a named rule (automatic | assisted | manual); assisted
    # and manual results must be routed to human review by the consumer.
    automation: str | None = None
    children: list["RuleResult"] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    ruleset: str
    version: int | None = None
    engine_version: str = ENGINE_VERSION
    extractor_version: str | None = None
    decision: Decision
    root: RuleResult
    # Each named rule that was actually evaluated, exactly once (DAG, not a
    # tree): rules referenced multiple times appear here once; rules never
    # reached (short-circuit) are honestly absent.
    rule_results: dict[str, RuleResult] = Field(default_factory=dict)
    facts: dict[str, FactInput]
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
