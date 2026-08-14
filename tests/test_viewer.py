"""Viewer: registry page, document rendering, evaluation overlay by path."""
import importlib

import pytest

from app.compiler import compile_document
from app.evaluator import evaluate_document
from app.models import AuthoringDocument, coerce_facts
from app.viewer import build_result_index, collect_fact_names


def cmp_(fact, op="eq", value=True):
    return {"kind": "comparison", "fact": fact, "operator": op, "value": value}


def ref(rule_id):
    return {"kind": "rule_ref", "rule": rule_id}


LIB = {
    "name": "common-kyc",
    "rules": [
        {"id": "adult", "label": "Customer is of age",
         "root": cmp_("age", "ge", 18)},
        {"id": "resident", "root": cmp_("residence", "eq", "Germany")},
        {"id": "complete", "label": "KYC complete",
         "root": {"kind": "all", "children": [ref("adult"), ref("resident")]}},
    ],
    "root": ref("complete"),
}

APP_DOC = {
    "name": "consumer-credit",
    "imports": [{"ruleset": "common-kyc"}],
    "rules": [
        {"id": "high_income_debt_free", "label": "Debt-free for high salaries",
         "root": {"kind": "conditional_requirement",
                  "when": cmp_("salary", "ge", 50_000),
                  "require": cmp_("debt_free", "eq", True)}},
    ],
    "root": {"kind": "all", "children": [
        ref("common-kyc:complete"), ref("high_income_debt_free")]},
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    from app import storage, main
    importlib.reload(storage)
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        c.post("/rulesets", json=LIB)
        c.post("/rulesets", json=APP_DOC)
        yield c


class TestPages:
    def test_registry_lists_rulesets(self, client):
        r = client.get("/ui")
        assert r.status_code == 200
        assert "common-kyc" in r.text and "consumer-credit" in r.text

    def test_document_renders_rules_imports_and_facts_skeleton(self, client):
        r = client.get("/ui/consumer-credit")
        assert r.status_code == 200
        # imported + local rules rendered, pinned import shown
        assert "common-kyc:adult" in r.text
        assert "Debt-free for high salaries" in r.text
        assert "Imports (pinned)" in r.text
        # facts skeleton contains every fact the rules read (incl. library)
        for fact in ("age", "residence", "salary", "debt_free"):
            assert f"&quot;{fact}&quot;" in r.text
        # no evaluation yet -> no status chips
        assert "chip s-satisfied" not in r.text

    def test_unknown_ruleset_404(self, client):
        assert client.get("/ui/nope").status_code == 404

    def test_draft_only_ruleset_needs_explicit_version(self, client):
        client.post("/rulesets?draft=true", json={
            "name": "draft-only", "root": cmp_("x")})
        assert client.get("/ui/draft-only").status_code == 404
        assert client.get("/ui/draft-only?version=1").status_code == 200


class TestEvaluationOverlay:
    def test_statuses_and_decision_rendered(self, client):
        r = client.post(
            "/ui/consumer-credit/evaluate?version=1",
            data={"facts_json":
                  '{"age": 34, "residence": "Germany", "salary": 45000}'},
        )
        assert r.status_code == 200
        assert "Eligible" in r.text
        assert "chip s-satisfied" in r.text
        assert "chip s-not_applicable" in r.text  # the gate

    def test_null_facts_evaluate_as_unknown(self, client):
        r = client.post(
            "/ui/consumer-credit/evaluate?version=1",
            data={"facts_json": '{"age": 34, "residence": null}'},
        )
        assert "Needs review" in r.text
        assert "chip s-unknown" in r.text

    def test_lazy_rules_render_as_not_evaluated(self, client):
        r = client.post(
            "/ui/consumer-credit/evaluate?version=1",
            data={"facts_json": '{"age": 15, "residence": "Germany"}'},
        )
        assert "Not eligible" in r.text
        assert "not evaluated" in r.text

    def test_invalid_json_shows_error_not_500(self, client):
        r = client.post(
            "/ui/consumer-credit/evaluate?version=1",
            data={"facts_json": "{not json"},
        )
        assert r.status_code == 200
        assert "Invalid facts" in r.text

    def test_ui_evaluations_land_in_audit_trail(self, client):
        from app import storage
        client.post(
            "/ui/consumer-credit/evaluate?version=1",
            data={"facts_json": '{"age": 34, "residence": "Germany"}'},
        )
        with storage.SessionLocal() as s:
            from sqlalchemy import func, select
            n = s.scalar(select(func.count(storage.EvaluationRecord.id)))
        assert n == 1


class TestPathContract:
    """The viewer relies on the evaluator's path convention — verify every
    path in a report resolves, and rendering covers the same paths."""

    def test_every_report_path_is_unique_and_indexable(self):
        lib = AuthoringDocument.model_validate(LIB)
        report = evaluate_document(
            lib, coerce_facts({"age": 34, "residence": "Germany"}))
        index = build_result_index(report)
        assert "root" in index
        assert "rule:complete" in index
        assert "rule:complete.children[0]" in index

    def test_collect_fact_names_spans_imports(self):
        lib = AuthoringDocument.model_validate(LIB)
        app_doc = AuthoringDocument.model_validate(APP_DOC)
        compiled = compile_document(
            app_doc, lambda name, v: (lib, 1))
        assert collect_fact_names(compiled) == [
            "age", "debt_free", "residence", "salary"]
