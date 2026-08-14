"""Reproduces the deck's 'Worked examples: how the decision resolves' table.

Rule: eligible when ALL of
  - age >= 28
  - residence == Germany
  - WHEN salary >= 50_000 REQUIRE debt_free == True
"""
import pytest

from app.evaluator import evaluate_document
from app.models import AuthoringDocument, coerce_facts
from app.report import Decision, RuleStatus

DOC = AuthoringDocument.model_validate(
    {
        "name": "retail-eligibility",
        "root": {
            "kind": "all",
            "label": "Customer is eligible when ALL of the following",
            "children": [
                {"kind": "comparison", "fact": "age", "operator": "ge",
                 "value": 28, "label": "Age is at least 28"},
                {"kind": "comparison", "fact": "residence", "operator": "eq",
                 "value": "Germany", "label": "Country of residence is Germany"},
                {
                    "kind": "conditional_requirement",
                    "label": "Debt-free for higher salaries",
                    "when": {"kind": "comparison", "fact": "salary",
                             "operator": "ge", "value": 50_000},
                    "require": {"kind": "comparison", "fact": "debt_free",
                                "operator": "eq", "value": True},
                },
            ],
        },
    }
)

UNKNOWN = {"status": "unknown"}

ROWS = [
    # age, residence, salary, debt_free, expected gate status, expected decision
    (34, "Germany", 45_000, None, RuleStatus.NOT_APPLICABLE, Decision.ELIGIBLE),
    (34, "Germany", 60_000, True, RuleStatus.SATISFIED, Decision.ELIGIBLE),
    (34, "Germany", 60_000, False, RuleStatus.FAILED, Decision.NOT_ELIGIBLE),
    (34, "Germany", UNKNOWN, True, RuleStatus.UNKNOWN, Decision.NEEDS_REVIEW),
    (24, "Germany", UNKNOWN, UNKNOWN, RuleStatus.NOT_EVALUATED,
     Decision.NOT_ELIGIBLE),
]


@pytest.mark.parametrize("age,residence,salary,debt_free,gate,decision", ROWS)
def test_worked_examples(age, residence, salary, debt_free, gate, decision):
    raw = {"age": age, "residence": residence}
    if salary is not None:
        raw["salary"] = salary
    if debt_free is not None:
        raw["debt_free"] = debt_free

    report = evaluate_document(DOC, coerce_facts(raw))

    assert report.decision is decision
    gate_result = report.root.children[2]
    assert gate_result.status is gate


def test_gate_separates_applicability_from_truth():
    report = evaluate_document(
        DOC, coerce_facts({"age": 34, "residence": "Germany",
                           "salary": 60_000, "debt_free": False})
    )
    gate = report.root.children[2]
    assert gate.applicability == "applicable"
    assert gate.truth == "false"

    report = evaluate_document(
        DOC, coerce_facts({"age": 34, "residence": "Germany",
                           "salary": 45_000})
    )
    gate = report.root.children[2]
    assert gate.applicability == "not_applicable"
    assert gate.truth is None  # truth undefined when not applicable

    # "does not apply" vs "unknown whether it applies" are NOT equivalent
    report = evaluate_document(
        DOC, coerce_facts({"age": 34, "residence": "Germany",
                           "salary": {"status": "unknown"}})
    )
    gate = report.root.children[2]
    assert gate.applicability == "unknown"
    assert gate.status is RuleStatus.UNKNOWN


def test_short_circuit_records_not_evaluated():
    """Row 5: a known failed age forces Not eligible; the debt-free branch is
    recorded as not_evaluated — the explanation stays honest."""
    report = evaluate_document(
        DOC, coerce_facts({"age": 24, "residence": "Germany"})
    )
    assert report.decision is Decision.NOT_ELIGIBLE
    assert report.root.children[0].status is RuleStatus.FAILED
    assert report.root.children[2].status is RuleStatus.NOT_EVALUATED
