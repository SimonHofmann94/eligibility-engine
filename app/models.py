"""Authoring model — the small, business-facing block vocabulary.

This is the deck's "AuthoringDocument": human-readable, nested, persisted as
the source of truth. Pydantic models are the single source of truth for both
the API contract and internal logic (deck: "Typed Pydantic models").

Lean-core vocabulary: comparison, all, any, not, conditional_requirement.
Extension points (calculation, aggregate, decision_table, quantifier) plug in
as new members of the `Node` discriminated union — see README.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

class FactStatus(str, Enum):
    """Every fact carries an explicit status — missing is never false."""

    KNOWN = "known"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


class FactInput(BaseModel):
    """A fact as supplied to an evaluation."""

    model_config = ConfigDict(extra="forbid")

    status: FactStatus = FactStatus.KNOWN
    value: Any = None
    # Optional resolution note — provenance or why the fact is unknown
    # (e.g. "ambiguous: conflicting extracted values"). Flows into reasons.
    note: str | None = None

    @model_validator(mode="after")
    def _known_needs_value(self) -> "FactInput":
        if self.status is FactStatus.KNOWN and self.value is None:
            raise ValueError("a fact with status 'known' must carry a value")
        return self


FactPayload = dict[str, FactInput]


def coerce_facts(raw: dict[str, Any]) -> FactPayload:
    """Accept both shorthand (``{"age": 34}``) and explicit form
    (``{"salary": {"status": "unknown"}}``).

    POC caveat: a raw dict value containing 'status' or 'value' keys is
    interpreted as the explicit form.
    """
    facts: FactPayload = {}
    for name, v in raw.items():
        if isinstance(v, dict) and ("status" in v or "value" in v):
            facts[name] = FactInput.model_validate(v)
        else:
            facts[name] = FactInput(value=v)
    return facts


# ---------------------------------------------------------------------------
# Rule blocks (discriminated on `kind`)
# ---------------------------------------------------------------------------

class _BaseBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str | None = None


Operator = Literal["eq", "ne", "gt", "ge", "lt", "le", "in", "not_in"]


class Comparison(_BaseBlock):
    """'Age is at least 28' — a typed comparison against one fact."""

    kind: Literal["comparison"] = "comparison"
    fact: str
    operator: Operator = "eq"
    value: Any


class AllBlock(_BaseBlock):
    """'All of the following' — Boolean aggregation."""

    kind: Literal["all"] = "all"
    children: list[Node] = Field(min_length=1)


class AnyBlock(_BaseBlock):
    """'At least one of the following' — Boolean aggregation."""

    kind: Literal["any"] = "any"
    children: list[Node] = Field(min_length=1)


class NotBlock(_BaseBlock):
    """'It is not the case that' — negation."""

    kind: Literal["not"] = "not"
    child: Node


class ConditionalRequirement(_BaseBlock):
    """'When X applies, require Y' — the applicability gate.

    NOT a plain implication: when `when` is conclusively false the block is
    not_applicable rather than trivially true.
    """

    kind: Literal["conditional_requirement"] = "conditional_requirement"
    when: Node
    require: Node


class OneOfBlock(_BaseBlock):
    """'Exactly one of the following' — e.g. income proved by exactly one
    alternative route. More than one satisfied alternative is a conflict.

    Deviation from the Konzept (documented): one satisfied child plus an
    unknown child yields UNKNOWN here, not satisfied — the unknown child
    could later resolve to satisfied and flip the result into a conflict,
    so the conservative reading avoids a false green.
    """

    kind: Literal["one_of"] = "one_of"
    children: list[Node] = Field(min_length=1)


class RuleRef(_BaseBlock):
    """'The outcome of rule X' — a reference to a named rule in the same
    document.

    References are how a rule depends on the outcome of another rule.
    Dependencies are discovered from references at compile time — authors
    never wire them by hand — and cycles are a compile-time error, never a
    runtime surprise.
    """

    kind: Literal["rule_ref"] = "rule_ref"
    rule: str = Field(min_length=1)


Node = Annotated[
    Union[Comparison, AllBlock, AnyBlock, NotBlock, ConditionalRequirement,
          OneOfBlock, RuleRef],
    Field(discriminator="kind"),
]

# Resolve the recursive forward references.
AllBlock.model_rebuild()
AnyBlock.model_rebuild()
NotBlock.model_rebuild()
ConditionalRequirement.model_rebuild()
OneOfBlock.model_rebuild()


class NamedRule(BaseModel):
    """A named, individually addressable rule within a document.

    Real rule systems (50–100 rules per credit process) are maintained as a
    flat catalog of named rules rather than one giant tree; the decision root
    and other rules refer to them via `rule_ref`.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")
    label: str | None = None
    description: str | None = None
    # Automatikgrad: 'automatic' runs unattended; 'assisted' and 'manual'
    # results must be routed to human review by the consumer of the report.
    automation: Literal["automatic", "assisted", "manual"] = "automatic"
    root: Node


class ImportSpec(BaseModel):
    """An import of another rule set's named rules.

    `version` may be omitted when submitting a document — the compiler then
    resolves the latest *published* version and the pin is frozen into the
    stored document at publish time ("pin-on-publish"), so evaluation is
    always replayable against exact versions. Only published versions are
    importable.
    """

    model_config = ConfigDict(extra="forbid")

    ruleset: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")
    version: int | None = Field(default=None, ge=1)


class AuthoringDocument(BaseModel):
    """The persisted source of truth for one rule set.

    `rules` is the catalog of named rules; `root` is the decision that ties
    them together (and may still contain inline blocks — small documents
    without named rules keep working unchanged). `imports` makes another
    rule set's named rules available under the namespace
    ``<ruleset>:<rule_id>`` — the shared-library mechanism across processes.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_\-]+$")
    description: str | None = None
    imports: list[ImportSpec] = Field(default_factory=list)
    rules: list[NamedRule] = Field(default_factory=list)
    root: Node

    @model_validator(mode="after")
    def _unique_rule_ids(self) -> "AuthoringDocument":
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id '{rule.id}'")
            seen.add(rule.id)
        return self
