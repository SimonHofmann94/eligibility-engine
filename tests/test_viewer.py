"""Viewer: registry page, document rendering, evaluation overlay by path."""
import importlib
import json

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


class TestAuthoring:
    def test_new_page_renders_starter(self, client):
        r = client.get("/ui/new")
        assert r.status_code == 200
        assert "my-ruleset" in r.text

    def test_edit_prefills_existing_version(self, client):
        r = client.get("/ui/new?from=consumer-credit&version=1")
        assert r.status_code == 200
        assert "high_income_debt_free" in r.text

    def test_edit_unknown_ruleset_404(self, client):
        assert client.get("/ui/new?from=nope").status_code == 404

    def test_validate_previews_without_persisting(self, client):
        doc = dict(LIB, name="preview-only")
        r = client.post("/ui/new", data={
            "doc_json": json.dumps(doc), "action": "validate"})
        assert r.status_code == 200
        assert "Preview" in r.text
        assert "KYC complete" in r.text
        assert client.get("/rulesets/preview-only/versions").status_code == 404

    def test_compile_error_surfaces(self, client):
        doc = {"name": "bad-ref", "root": ref("missing")}
        r = client.post("/ui/new", data={
            "doc_json": json.dumps(doc), "action": "publish"})
        assert r.status_code == 200
        assert "Compile errors" in r.text
        assert client.get("/rulesets/bad-ref/versions").status_code == 404

    def test_malformed_json_shows_error_not_500(self, client):
        r = client.post("/ui/new",
                        data={"doc_json": "{nope", "action": "publish"})
        assert r.status_code == 200
        assert "Invalid document" in r.text

    def test_publish_creates_version_and_redirects(self, client):
        doc = {"name": "ui-born", "root": cmp_("x")}
        r = client.post("/ui/new",
                        data={"doc_json": json.dumps(doc),
                              "action": "publish"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/ui/ui-born?version=1"
        assert client.get("/ui/ui-born").status_code == 200

    def test_publish_draft_is_not_latest(self, client):
        doc = {"name": "ui-draft", "root": cmp_("x")}
        client.post("/ui/new", data={
            "doc_json": json.dumps(doc), "action": "publish", "draft": "1"},
            follow_redirects=False)
        assert client.get("/ui/ui-draft").status_code == 404
        assert client.get("/ui/ui-draft?version=1").status_code == 200

    def test_publish_pins_imports(self, client):
        doc = {"name": "ui-pins", "imports": [{"ruleset": "common-kyc"}],
               "root": ref("common-kyc:complete")}
        client.post("/ui/new",
                    data={"doc_json": json.dumps(doc), "action": "publish"})
        stored = client.get("/rulesets/ui-pins/versions/1").json()
        assert stored["imports"][0]["version"] == 1


class TestFactForm:
    ELIGIBLE_ROWS = {
        "f:age": "34", "s:age": "known",
        "f:residence": "Germany", "s:residence": "known",
        "f:salary": "45000", "s:salary": "known",
    }

    def test_fact_rows_rendered(self, client):
        r = client.get("/ui/consumer-credit")
        for fact in ("age", "residence", "salary", "debt_free"):
            assert f"name='f:{fact}'" in r.text

    def test_form_rows_evaluate(self, client):
        r = client.post("/ui/consumer-credit/evaluate?version=1",
                        data=self.ELIGIBLE_ROWS)
        assert "Not eligible" not in r.text
        assert "Eligible" in r.text
        assert "chip s-not_applicable" in r.text  # the gate

    def test_empty_known_row_is_not_provided(self, client):
        data = dict(self.ELIGIBLE_ROWS, **{"f:residence": ""})
        r = client.post("/ui/consumer-credit/evaluate?version=1", data=data)
        assert "Needs review" in r.text

    def test_status_select_respected(self, client):
        data = dict(self.ELIGIBLE_ROWS, **{"s:age": "unknown"})
        r = client.post("/ui/consumer-credit/evaluate?version=1", data=data)
        assert "Needs review" in r.text

    def test_row_wins_over_facts_json(self, client):
        r = client.post(
            "/ui/consumer-credit/evaluate?version=1",
            data={"facts_json":
                  '{"age": 15, "residence": "Germany", "salary": 45000}',
                  "f:age": "34", "s:age": "known"},
        )
        assert "Not eligible" not in r.text
        assert "Eligible" in r.text

    def test_statements_below_threshold_resolve_unknown(self, client):
        stmts = [{"field": "age", "value": 34, "confidence": 0.5}]
        r = client.post(
            "/ui/consumer-credit/evaluate?version=1",
            data={"statements_json": json.dumps(stmts),
                  "confidence_threshold": "0.8",
                  "f:residence": "Germany", "s:residence": "known"},
        )
        assert "Needs review" in r.text
        assert "chip s-unknown" in r.text

    def test_statements_above_threshold_used(self, client):
        stmts = [
            {"field": "age", "value": 34, "confidence": 0.95},
            {"field": "residence", "value": "Germany", "confidence": 0.9},
            {"field": "salary", "value": 45000, "confidence": 0.9},
        ]
        r = client.post(
            "/ui/consumer-credit/evaluate?version=1",
            data={"statements_json": json.dumps(stmts)},
        )
        assert "Not eligible" not in r.text
        assert "Eligible" in r.text


class TestLifecycle:
    def test_draft_shows_promote_button(self, client):
        client.post("/rulesets?draft=true",
                    json={"name": "d1", "root": cmp_("x")})
        r = client.get("/ui/d1?version=1")
        assert "Promote to published" in r.text

    def test_promote_flow(self, client):
        client.post("/rulesets?draft=true",
                    json={"name": "d2", "root": cmp_("x")})
        r = client.post("/ui/d2/promote?version=1", follow_redirects=False)
        assert r.status_code == 303
        r2 = client.get("/ui/d2")  # latest published now resolves
        assert r2.status_code == 200
        assert "Promote to published" not in r2.text

    def test_promote_published_409(self, client):
        r = client.post("/ui/consumer-credit/promote?version=1")
        assert r.status_code == 409

    def test_promote_unknown_404(self, client):
        assert client.post("/ui/nope/promote?version=1").status_code == 404


class TestHistory:
    def test_empty_state(self, client):
        r = client.get("/ui/consumer-credit/history")
        assert r.status_code == 200
        assert "No evaluations yet" in r.text

    def test_unknown_name_404(self, client):
        assert client.get("/ui/nope/history").status_code == 404

    def test_rows_newest_first_and_detail_replays(self, client):
        client.post("/ui/consumer-credit/evaluate?version=1",
                    data={"facts_json":
                          '{"age": 15, "residence": "Germany"}'})
        client.post("/ui/consumer-credit/evaluate?version=1",
                    data={"facts_json": '{"age": 34, "residence": "Germany",'
                          ' "salary": 45000}'})
        r = client.get("/ui/consumer-credit/history")
        assert "Not eligible" in r.text and "Eligible" in r.text
        assert r.text.index("/evaluations/2") < r.text.index("/evaluations/1")

        d = client.get("/ui/consumer-credit/evaluations/2")
        assert d.status_code == 200
        assert "Historical evaluation" in d.text
        assert "chip s-satisfied" in d.text

    def test_unknown_id_404(self, client):
        r = client.get("/ui/consumer-credit/evaluations/999")
        assert r.status_code == 404

    def test_cross_ruleset_id_404(self, client):
        client.post("/ui/consumer-credit/evaluate?version=1",
                    data={"facts_json": '{"age": 34}'})
        assert client.get("/ui/common-kyc/evaluations/1").status_code == 404


class TestDiff:
    def test_diff_marks_changes(self, client):
        lib2 = json.loads(json.dumps(LIB))
        lib2["rules"][0]["root"]["value"] = 21
        client.post("/rulesets", json=lib2)
        r = client.get("/ui/common-kyc/diff?a=1&b=2")
        assert r.status_code == 200
        assert "ddel" in r.text and "dadd" in r.text
        assert "21" in r.text

    def test_same_version_no_differences(self, client):
        r = client.get("/ui/common-kyc/diff?a=1&b=1")
        assert "No differences" in r.text

    def test_missing_version_404(self, client):
        assert client.get("/ui/common-kyc/diff?a=1&b=9").status_code == 404
