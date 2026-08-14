"""Statement layer — extracted Aussagen and their resolution into facts.

Implements the Konzept's Stufe-0/Stufe-1 boundary on the evaluation side:
extraction systems (LLM, OCR, NER) produce *statements* — value claims with
provenance and confidence. Contradictory statements about the same field are
allowed and wanted; they are the data basis of consistency checking. The
engine itself never sees statements: this module resolves them into the
FactPayload the evaluator already consumes, so the four-state propagation
downstream needs no changes.

Two resolvers, per the Konzept:

* ``resolve_value``  — one value for conditions and calculations.
  Precedence: override > highest confidence > most recent. Below-threshold
  confidence and unresolved ambiguity yield an *unknown* fact (never a
  guess), with the reason carried in the fact's note.
* ``resolve_set``    — the unresolved set of distinct values, for
  consistency rules ("same value everywhere").
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import FactInput, FactPayload, FactStatus

DEFAULT_CONFIDENCE_THRESHOLD = 0.8


class Statement(BaseModel):
    """A single extracted or overridden claim about one field.

    Mirrors the Konzept's ExtractedField tuple: (field, value, source,
    location, confidence, extractor version). Overrides are statements too —
    highest precedence, never overwritten, always attributed (who/when/why).

    Subject scoping (claims about Person_A vs Person_B) arrives with the
    entity-relationship step; until then fields live in the flat fact
    namespace the engine uses today.
    """

    model_config = ConfigDict(extra="forbid")

    field: str = Field(min_length=1)
    value: Any
    kind: Literal["extracted", "override"] = "extracted"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    stated_at: datetime | None = None
    source_document: str | None = None
    source_location: dict | None = None
    extractor_version: str | None = None
    # Override attribution — required for overrides
    author: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _override_needs_author(self) -> "Statement":
        if self.kind == "override" and not self.author:
            raise ValueError("an override statement must name its author")
        return self


def _eligible(stmts: list[Statement], threshold: float) -> list[Statement]:
    return [
        s for s in stmts
        if s.kind == "extracted"
        and (s.confidence is None or s.confidence >= threshold)
    ]


def _ts(s: Statement) -> datetime:
    return s.stated_at or datetime.min


def resolve_value(
    stmts: list[Statement],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> FactInput:
    """Collapse all statements about one field into a single fact.

    Override > highest confidence > most recent; ambiguity and
    below-threshold-only evidence resolve to unknown, never to a guess.
    """
    overrides = [s for s in stmts if s.kind == "override"]
    if overrides:
        o = max(overrides, key=_ts)
        return FactInput(
            value=o.value,
            note=f"override by {o.author}"
            + (f": {o.reason}" if o.reason else ""),
        )

    eligible = _eligible(stmts, threshold)
    if not eligible:
        if stmts:
            best = max(
                (s.confidence or 0.0) for s in stmts if s.kind == "extracted"
            )
            return FactInput(
                status=FactStatus.UNKNOWN,
                note=f"all extracted values below confidence threshold "
                f"({best:.2f} < {threshold:.2f})",
            )
        return FactInput(status=FactStatus.UNKNOWN, note="no statements")

    best_conf = max(s.confidence if s.confidence is not None else 1.0
                    for s in eligible)
    top = [s for s in eligible
           if (s.confidence if s.confidence is not None else 1.0) == best_conf]

    distinct = {repr(s.value) for s in top}
    if len(distinct) == 1:
        s = top[0]
        return FactInput(value=s.value, note=_provenance(s))

    # Same confidence, different values -> recency decides, if it can.
    newest = max(top, key=_ts)
    tied_newest = [s for s in top if _ts(s) == _ts(newest)]
    if len({repr(s.value) for s in tied_newest}) == 1:
        return FactInput(value=newest.value,
                         note=_provenance(newest) + ", most recent")

    values = sorted({repr(s.value) for s in tied_newest})
    return FactInput(
        status=FactStatus.UNKNOWN,
        note="ambiguous: equally confident conflicting values "
        + " vs ".join(values) + " — human review required",
    )


def resolve_set(
    stmts: list[Statement],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[Any]:
    """The unresolved set of distinct, sufficiently confident values —
    the data basis for consistency rules. Two or more entries mean the
    'same value everywhere' rule is violated."""
    seen: list[Any] = []
    for s in _eligible(stmts, threshold):
        if all(repr(s.value) != repr(v) for v in seen):
            seen.append(s.value)
    return seen


def resolve_facts(
    statements: list[Statement],
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> FactPayload:
    """Group statements by field and resolve each into a fact."""
    by_field: dict[str, list[Statement]] = {}
    for s in statements:
        by_field.setdefault(s.field, []).append(s)
    return {
        field: resolve_value(group, threshold)
        for field, group in by_field.items()
    }


def _provenance(s: Statement) -> str:
    parts = []
    if s.source_document:
        loc = ""
        if s.source_location and "page" in s.source_location:
            loc = f" p.{s.source_location['page']}"
        parts.append(f"from {s.source_document}{loc}")
    if s.confidence is not None:
        parts.append(f"confidence {s.confidence:.2f}")
    return ", ".join(parts) or "extracted"
