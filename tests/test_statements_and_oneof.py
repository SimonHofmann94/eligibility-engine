"""v0.5: one_of semantics, statement layer resolvers, version/automation stamps."""
import importlib
from datetime import datetime

import pytest

from app.evaluator import Evaluator, evaluate_document
from app.models import AuthoringDocument, FactStatus, coerce_facts
from app.report import ENGINE_VERSION, RuleStatus
from app.statements import Statement, resolve_facts, resolve_set, resolve_value


def cmp_(fact, op="eq", value=True):
    return {"kind": "comparison", "fact": fact, "operator": op, "value": value}


def one_of(*children):
    return AuthoringDocument.model_validate(
        {"name": "t", "root": {"kind": "one_of", "children": list(children)}})


def status_of(doc, facts) -> RuleStatus:
    return Evaluator(coerce_facts(facts)).evaluate(doc)


GATE = {"kind": "conditional_requirement",
        "when": cmp_("gate"), "require": cmp_("req")}


class TestOneOf:
    DOC = one_of(cmp_("a"), cmp_("b"), cmp_("c"))

    def test_exactly_one_satisfied(self):
        r = status_of(self.DOC, {"a": True, "b": False, "c": False})
        assert r.status is RuleStatus.SATISFIED

    def test_two_satisfied_is_a_conflict(self):
        r = status_of(self.DOC, {"a": True, "b": True, "c": False})
        assert r.status is RuleStatus.FAILED
        assert "conflict" in r.reason

    def test_none_satisfied_fails(self):
        r = status_of(self.DOC, {"a": False, "b": False, "c": False})
        assert r.status is RuleStatus.FAILED

    def test_conservative_deviation_one_green_plus_unknown_is_unknown(self):
        # Konzept table would say satisfied; we deviate deliberately:
        r = status_of(self.DOC, {"a": True, "b": False})  # c unknown
        assert r.status is RuleStatus.UNKNOWN

    def test_conflict_beats_unknown(self):
        r = status_of(self.DOC, {"a": True, "b": True})  # c unknown
        assert r.status is RuleStatus.FAILED

    def test_all_not_applicable(self):
        doc = one_of(GATE)
        r = status_of(doc, {"gate": False})
        assert r.status is RuleStatus.NOT_APPLICABLE


class TestResolveValue:
    def test_single_confident_statement(self):
        f = resolve_value([Statement(field="name", value="Müller",
                                     confidence=0.94,
                                     source_document="doc_abc")])
        assert f.status is FactStatus.KNOWN and f.value == "Müller"
        assert "doc_abc" in f.note

    def test_override_beats_everything(self):
        f = resolve_value([
            Statement(field="name", value="Miller", confidence=0.99),
            Statement(field="name", value="Müller", kind="override",
                      author="j.schmidt", reason="checked against ID"),
        ])
        assert f.value == "Müller"
        assert "override by j.schmidt" in f.note

    def test_override_requires_author(self):
        with pytest.raises(Exception, match="author"):
            Statement(field="x", value=1, kind="override")

    def test_higher_confidence_wins(self):
        f = resolve_value([
            Statement(field="salary", value=45_000, confidence=0.6),
            Statement(field="salary", value=52_000, confidence=0.95),
        ])
        assert f.value == 52_000

    def test_below_threshold_is_unknown_never_a_guess(self):
        f = resolve_value([Statement(field="iban", value="DE...",
                                     confidence=0.5)])
        assert f.status is FactStatus.UNKNOWN
        assert "below confidence threshold" in f.note

    def test_recency_breaks_confidence_ties(self):
        f = resolve_value([
            Statement(field="address", value="Old Str. 1", confidence=0.9,
                      stated_at=datetime(2026, 1, 1)),
            Statement(field="address", value="New Str. 2", confidence=0.9,
                      stated_at=datetime(2026, 6, 1)),
        ])
        assert f.value == "New Str. 2"
        assert "most recent" in f.note

    def test_true_ambiguity_is_unknown_with_hitl_note(self):
        f = resolve_value([
            Statement(field="price", value=300_000, confidence=0.9),
            Statement(field="price", value=310_000, confidence=0.9),
        ])
        assert f.status is FactStatus.UNKNOWN
        assert "human review required" in f.note

    def test_resolved_unknown_note_reaches_the_report(self):
        doc = AuthoringDocument.model_validate(
            {"name": "t", "root": cmp_("price", "ge", 100_000)})
        facts = resolve_facts([
            Statement(field="price", value=300_000, confidence=0.9),
            Statement(field="price", value=310_000, confidence=0.9),
        ])
        report = evaluate_document(doc, facts)
        assert "human review required" in report.root.reason


class TestResolveSet:
    def test_distinct_confident_values(self):
        s = resolve_set([
            Statement(field="name", value="Müller", confidence=0.94),
            Statement(field="name", value="Mueller", confidence=0.91),
            Statement(field="name", value="Müller", confidence=0.99),
            Statement(field="name", value="Typo", confidence=0.3),  # dropped
        ])
        assert s == ["Müller", "Mueller"]  # 2 entries -> inconsistency


class TestApiV05:
    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
        from app import storage, main
        importlib.reload(storage)
        importlib.reload(main)
        from fastapi.testclient import TestClient
        with TestClient(main.app) as c:
            c.post("/rulesets", json={
                "name": "demo",
                "rules": [{"id": "income_ok", "automation": "assisted",
                           "root": cmp_("salary", "ge", 50_000)}],
                "root": {"kind": "rule_ref", "rule": "income_ok"},
            })
            yield c

    def test_statements_payload_end_to_end(self, client):
        r = client.post("/rulesets/demo/evaluate", json={
            "statements": [
                {"field": "salary", "value": 60_000, "confidence": 0.93,
                 "source_document": "gehaltsabrechnung_03",
                 "extractor_version": "extractor-v2.1"},
            ],
        })
        body = r.json()
        assert body["decision"] == "eligible"
        assert body["engine_version"] == ENGINE_VERSION
        assert body["extractor_version"] == "extractor-v2.1"
        assert body["rule_results"]["income_ok"]["automation"] == "assisted"
        assert "gehaltsabrechnung_03" in \
            body["rule_results"]["income_ok"]["reason"]

    def test_explicit_facts_win_over_statements(self, client):
        r = client.post("/rulesets/demo/evaluate", json={
            "statements": [{"field": "salary", "value": 10, "confidence": 0.99}],
            "facts": {"salary": 60_000},
        })
        assert r.json()["decision"] == "eligible"

    def test_neither_facts_nor_statements_rejected(self, client):
        assert client.post("/rulesets/demo/evaluate", json={}).status_code == 422
