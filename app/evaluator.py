"""The evaluation engine — four-state semantics with applicability gates.

Implements the deck's semantics precisely:

* comparison  -> typed comparison; unknown input propagates as unknown,
                 never guessed as false; type mismatch is an error.
* all         -> failed > error > unknown > all-not_applicable > satisfied
* any         -> satisfied > error > unknown > all-not_applicable > failed
* not         -> flips satisfied/failed; unknown, not_applicable and error
                 pass through.
* conditional_requirement (WHEN A REQUIRE B):
      A satisfied      -> evaluate B (B's status carries through)
      A failed         -> not_applicable
      A unknown        -> unknown   ("unknown whether the rule applies")
      A not_applicable -> not_applicable
      A error          -> error

Short-circuiting: once a group's outcome is determined (a failed child in
`all`, a satisfied child in `any`), remaining children are recorded as
`not_evaluated` — an execution state, not a truth status — so the explanation
stays honest about what was and was not checked.

Note on error precedence in groups: the deck leaves this unspecified. We rank
error directly after the decisive state (failed for `all`, satisfied for
`any`) so infrastructure problems are never masked by mere unknowns.

Note on a deck-internal tension: the fact-status slide says a not_applicable
*input* makes the requiring rule not_applicable, while the "important
fact-handling rule" slide says it evaluates as unknown ("not_applicable is
never inferred just because data is missing"). We follow the latter as the
default because it is the conservative choice, and expose
``na_input_policy="not_applicable"`` for the other reading.
"""
from __future__ import annotations

import operator as _op
from typing import Any, Callable, Literal

from .models import (
    AllBlock,
    AnyBlock,
    AuthoringDocument,
    Comparison,
    ConditionalRequirement,
    FactPayload,
    FactStatus,
    NotBlock,
    OneOfBlock,
    RuleRef,
)
from .report import (
    STATUS_TO_DECISION,
    Decision,
    EvaluationReport,
    RuleResult,
    RuleStatus,
)

_OPS: dict[str, tuple[str, Callable[[Any, Any], bool]]] = {
    "eq": ("=", _op.eq),
    "ne": ("≠", _op.ne),
    "gt": (">", _op.gt),
    "ge": ("≥", _op.ge),
    "lt": ("<", _op.lt),
    "le": ("≤", _op.le),
    "in": ("in", lambda a, b: a in b),
    "not_in": ("not in", lambda a, b: a not in b),
    "between": ("between", lambda a, b: b[0] <= a <= b[1]),
}


class Evaluator:
    def __init__(
        self,
        facts: FactPayload,
        rules_by_id: dict[str, Any] | None = None,
        *,
        short_circuit: bool = True,
        na_input_policy: Literal["unknown", "not_applicable"] = "unknown",
    ) -> None:
        self.facts = facts
        self.rules_by_id = rules_by_id or {}
        self.short_circuit = short_circuit
        self.na_input_policy = na_input_policy
        # Memo: each named rule is evaluated at most once per request, no
        # matter how many places reference it (lazy evaluation over a DAG).
        self._rule_memo: dict[str, RuleResult] = {}

    # -- public entry point -------------------------------------------------

    def evaluate(self, doc: AuthoringDocument) -> RuleResult:
        return self._eval(doc.root, "root")

    def report(
        self,
        doc: AuthoringDocument,
        version: int | None = None,
        extractor_version: str | None = None,
    ) -> EvaluationReport:
        root = self.evaluate(doc)
        return EvaluationReport(
            ruleset=doc.name,
            version=version,
            extractor_version=extractor_version,
            decision=STATUS_TO_DECISION[root.status],
            root=root,
            rule_results=dict(self._rule_memo),
            facts=self.facts,
        )

    # -- dispatch -----------------------------------------------------------

    def _eval(self, node: Any, path: str) -> RuleResult:
        if isinstance(node, Comparison):
            return self._comparison(node, path)
        if isinstance(node, AllBlock):
            return self._group(node, path, mode="all")
        if isinstance(node, AnyBlock):
            return self._group(node, path, mode="any")
        if isinstance(node, OneOfBlock):
            return self._one_of(node, path)
        if isinstance(node, NotBlock):
            return self._not(node, path)
        if isinstance(node, ConditionalRequirement):
            return self._conditional(node, path)
        if isinstance(node, RuleRef):
            return self._rule_ref(node, path)
        raise TypeError(f"unsupported node kind: {type(node).__name__}")

    # -- rule references ----------------------------------------------------

    def _rule_ref(self, node: RuleRef, path: str) -> RuleResult:
        rule = self.rules_by_id.get(node.rule)
        if rule is None:
            # Unreachable for compiled documents; kept as a guard for direct
            # engine use without the compile step.
            return RuleResult(
                path=path, kind=node.kind, label=node.label, ref=node.rule,
                status=RuleStatus.ERROR,
                reason=f"unresolved rule reference '{node.rule}' "
                "(document was not compiled)",
            )
        if node.rule not in self._rule_memo:
            result = self._eval(rule.root, f"rule:{node.rule}")
            result.automation = rule.automation
            self._rule_memo[node.rule] = result
        target = self._rule_memo[node.rule]
        return RuleResult(
            path=path, kind=node.kind,
            label=node.label or rule.label, ref=node.rule,
            status=target.status,
            reason=f"outcome of rule '{node.rule}' is {target.status.value}",
        )

    @staticmethod
    def _skipped(node: Any, path: str) -> RuleResult:
        return RuleResult(
            path=path,
            kind=node.kind,
            label=node.label,
            status=RuleStatus.NOT_EVALUATED,
            reason="not evaluated — outcome was already determined "
            "(short-circuit)",
        )

    # -- leaf: comparison ---------------------------------------------------

    def _comparison(self, node: Comparison, path: str) -> RuleResult:
        base = dict(path=path, kind=node.kind, label=node.label,
                    facts_read=[node.fact])
        fact = self.facts.get(node.fact)

        if fact is None or fact.status is FactStatus.UNKNOWN:
            note = f" — {fact.note}" if fact and fact.note else ""
            return RuleResult(
                **base,
                status=RuleStatus.UNKNOWN,
                reason=f"fact '{node.fact}' is unknown"
                + ("" if fact else " (not provided)") + note,
            )
        if fact.status is FactStatus.ERROR:
            return RuleResult(
                **base,
                status=RuleStatus.ERROR,
                reason=f"fact '{node.fact}' is in error state",
            )
        if fact.status is FactStatus.NOT_APPLICABLE:
            if self.na_input_policy == "unknown":
                return RuleResult(
                    **base,
                    status=RuleStatus.UNKNOWN,
                    reason=f"required input '{node.fact}' is not applicable",
                )
            return RuleResult(
                **base,
                status=RuleStatus.NOT_APPLICABLE,
                reason=f"input '{node.fact}' is not applicable",
            )

        symbol, fn = _OPS[node.operator]
        try:
            ok = fn(fact.value, node.value)
        except TypeError:
            return RuleResult(
                **base,
                status=RuleStatus.ERROR,
                reason=(
                    f"type mismatch comparing '{node.fact}' "
                    f"({type(fact.value).__name__}) {symbol} "
                    f"{node.value!r} ({type(node.value).__name__})"
                ),
            )
        note = f" [{fact.note}]" if fact.note else ""
        return RuleResult(
            **base,
            status=RuleStatus.SATISFIED if ok else RuleStatus.FAILED,
            reason=f"{node.fact} ({fact.value!r}) {symbol} {node.value!r} "
            f"is {'satisfied' if ok else 'not satisfied'}{note}",
        )

    # -- groups: all / any --------------------------------------------------

    def _group(self, node: AllBlock | AnyBlock, path: str,
               mode: Literal["all", "any"]) -> RuleResult:
        decisive = RuleStatus.FAILED if mode == "all" else RuleStatus.SATISFIED
        results: list[RuleResult] = []
        decided = False

        for i, child in enumerate(node.children):
            child_path = f"{path}.children[{i}]"
            if decided and self.short_circuit:
                results.append(self._skipped(child, child_path))
                continue
            r = self._eval(child, child_path)
            results.append(r)
            if r.status is decisive:
                decided = True

        seen = {r.status for r in results if r.status is not RuleStatus.NOT_EVALUATED}

        if decisive in seen:
            status, why = decisive, (
                "an applicable child failed" if mode == "all"
                else "at least one child is satisfied"
            )
        elif RuleStatus.ERROR in seen:
            status, why = RuleStatus.ERROR, "a child evaluation errored"
        elif RuleStatus.UNKNOWN in seen:
            status, why = RuleStatus.UNKNOWN, "at least one child is unknown"
        elif seen == {RuleStatus.NOT_APPLICABLE}:
            status, why = RuleStatus.NOT_APPLICABLE, "every child is not applicable"
        else:
            status, why = (
                (RuleStatus.SATISFIED, "every applicable child is satisfied")
                if mode == "all"
                else (RuleStatus.FAILED, "no child is satisfied")
            )

        return RuleResult(path=path, kind=node.kind, label=node.label,
                          status=status, reason=why, children=results)

    # -- one_of: exactly one alternative --------------------------------------

    def _one_of(self, node: OneOfBlock, path: str) -> RuleResult:
        results: list[RuleResult] = []
        satisfied = 0
        decided = False

        for i, child in enumerate(node.children):
            child_path = f"{path}.children[{i}]"
            if decided and self.short_circuit:
                results.append(self._skipped(child, child_path))
                continue
            r = self._eval(child, child_path)
            results.append(r)
            if r.status is RuleStatus.SATISFIED:
                satisfied += 1
                if satisfied >= 2:
                    decided = True  # conflict is certain, whatever remains

        seen = {r.status for r in results
                if r.status is not RuleStatus.NOT_EVALUATED}

        if satisfied >= 2:
            status, why = RuleStatus.FAILED, (
                "conflict: more than one alternative is satisfied")
        elif RuleStatus.ERROR in seen:
            status, why = RuleStatus.ERROR, "a child evaluation errored"
        elif RuleStatus.UNKNOWN in seen:
            # Conservative: an unknown child could resolve to satisfied and
            # flip the result into a conflict — never a false green.
            status, why = RuleStatus.UNKNOWN, (
                "at least one child is unknown — result could still flip")
        elif satisfied == 1:
            status, why = RuleStatus.SATISFIED, (
                "exactly one alternative is satisfied")
        elif seen == {RuleStatus.NOT_APPLICABLE}:
            status, why = RuleStatus.NOT_APPLICABLE, (
                "every child is not applicable")
        else:
            status, why = RuleStatus.FAILED, "no alternative is satisfied"

        return RuleResult(path=path, kind=node.kind, label=node.label,
                          status=status, reason=why, children=results)

    # -- not ----------------------------------------------------------------

    _NOT_MAP = {
        RuleStatus.SATISFIED: RuleStatus.FAILED,
        RuleStatus.FAILED: RuleStatus.SATISFIED,
        RuleStatus.UNKNOWN: RuleStatus.UNKNOWN,
        RuleStatus.NOT_APPLICABLE: RuleStatus.NOT_APPLICABLE,
        RuleStatus.ERROR: RuleStatus.ERROR,
    }

    def _not(self, node: NotBlock, path: str) -> RuleResult:
        child = self._eval(node.child, f"{path}.child")
        status = self._NOT_MAP[child.status]
        return RuleResult(
            path=path, kind=node.kind, label=node.label, status=status,
            reason=f"negation of child ({child.status.value})",
            children=[child],
        )

    # -- conditional requirement: WHEN A REQUIRE B --------------------------

    def _conditional(self, node: ConditionalRequirement, path: str) -> RuleResult:
        when = self._eval(node.when, f"{path}.when")
        base = dict(path=path, kind=node.kind, label=node.label)

        if when.status is RuleStatus.SATISFIED:
            require = self._eval(node.require, f"{path}.require")
            truth = {
                RuleStatus.SATISFIED: "true",
                RuleStatus.FAILED: "false",
                RuleStatus.UNKNOWN: "unknown",
            }.get(require.status)
            return RuleResult(
                **base,
                status=require.status,
                applicability="applicable",
                truth=truth,
                reason=f"applicability condition satisfied; requirement is "
                f"{require.status.value}",
                children=[when, require],
            )

        skipped = self._skipped(node.require, f"{path}.require")

        if when.status is RuleStatus.FAILED:
            status, appl, why = (
                RuleStatus.NOT_APPLICABLE, "not_applicable",
                "applicability condition conclusively false — "
                "requirement does not apply",
            )
        elif when.status is RuleStatus.UNKNOWN:
            status, appl, why = (
                RuleStatus.UNKNOWN, "unknown",
                "it is unknown whether the requirement applies",
            )
        elif when.status is RuleStatus.NOT_APPLICABLE:
            status, appl, why = (
                RuleStatus.NOT_APPLICABLE, "not_applicable",
                "applicability condition is itself not applicable",
            )
        else:  # ERROR
            status, appl, why = (
                RuleStatus.ERROR, None,
                "applicability condition errored",
            )

        return RuleResult(**base, status=status, applicability=appl,
                          reason=why, children=[when, skipped])


def evaluate_document(
    doc: AuthoringDocument,
    facts: FactPayload,
    *,
    version: int | None = None,
    short_circuit: bool = True,
    resolver=None,
    extractor_version: str | None = None,
) -> EvaluationReport:
    """Compile (resolve references and imports, check cycles) and evaluate."""
    from .compiler import compile_document

    compiled = compile_document(doc, resolver)
    return Evaluator(
        facts, compiled.rules_by_id, short_circuit=short_circuit
    ).report(doc, version, extractor_version=extractor_version)
