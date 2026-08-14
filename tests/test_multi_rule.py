"""Multi-rule documents: named rules, references, compiler guarantees."""
import pytest

from app.compiler import CompileError, compile_document
from app.evaluator import Evaluator, evaluate_document
from app.models import AuthoringDocument, coerce_facts
from app.report import Decision, RuleStatus


def cmp_(fact, op="eq", value=True):
    return {"kind": "comparison", "fact": fact, "operator": op, "value": value}


def ref(rule_id):
    return {"kind": "rule_ref", "rule": rule_id}


MULTI = AuthoringDocument.model_validate({
    "name": "consumer-credit-de",
    "rules": [
        {"id": "adult", "label": "Customer is of age",
         "root": cmp_("age", "ge", 18)},
        {"id": "resident", "label": "German residence",
         "root": cmp_("residence", "eq", "Germany")},
        # A rule depending on the outcome of other rules:
        {"id": "kyc_complete", "label": "KYC complete",
         "root": {"kind": "all", "children": [ref("adult"), ref("resident")]}},
        {"id": "high_income_debt_free",
         "root": {"kind": "conditional_requirement",
                  "when": cmp_("salary", "ge", 50_000),
                  "require": cmp_("debt_free", "eq", True)}},
    ],
    "root": {"kind": "all", "children": [
        ref("kyc_complete"),
        ref("adult"),            # referenced a second time on purpose
        ref("high_income_debt_free"),
    ]},
})


class TestCompiler:
    def test_compiles_with_dependency_order(self):
        compiled = compile_document(MULTI)
        order = compiled.topo_order
        assert order.index("adult") < order.index("kyc_complete")
        assert order.index("resident") < order.index("kyc_complete")
        assert compiled.dependencies["kyc_complete"] == {"adult", "resident"}

    def test_unknown_reference_is_a_compile_error_with_path(self):
        doc = AuthoringDocument.model_validate({
            "name": "t", "rules": [],
            "root": ref("does_not_exist"),
        })
        with pytest.raises(CompileError) as e:
            compile_document(doc)
        assert "unknown rule 'does_not_exist'" in e.value.errors[0]
        assert e.value.errors[0].startswith("root")

    def test_cycle_is_a_compile_error_never_a_runtime_surprise(self):
        doc = AuthoringDocument.model_validate({
            "name": "t",
            "rules": [
                {"id": "a", "root": ref("b")},
                {"id": "b", "root": ref("a")},
            ],
            "root": ref("a"),
        })
        with pytest.raises(CompileError) as e:
            compile_document(doc)
        assert any("circular" in msg for msg in e.value.errors)

    def test_self_reference_rejected(self):
        doc = AuthoringDocument.model_validate({
            "name": "t",
            "rules": [{"id": "a", "root": ref("a")}],
            "root": ref("a"),
        })
        with pytest.raises(CompileError) as e:
            compile_document(doc)
        assert "references itself" in e.value.errors[0]

    def test_all_errors_reported_at_once(self):
        doc = AuthoringDocument.model_validate({
            "name": "t",
            "rules": [{"id": "a", "root": ref("missing_1")}],
            "root": ref("missing_2"),
        })
        with pytest.raises(CompileError) as e:
            compile_document(doc)
        assert len(e.value.errors) == 2

    def test_duplicate_rule_ids_rejected_by_the_model(self):
        with pytest.raises(Exception, match="duplicate rule id"):
            AuthoringDocument.model_validate({
                "name": "t",
                "rules": [{"id": "a", "root": cmp_("x")},
                          {"id": "a", "root": cmp_("y")}],
                "root": ref("a"),
            })


class TestMultiRuleEvaluation:
    FACTS = {"age": 34, "residence": "Germany", "salary": 45_000}

    def test_end_to_end_decision(self):
        report = evaluate_document(MULTI, coerce_facts(self.FACTS))
        assert report.decision is Decision.ELIGIBLE
        # Every evaluated rule appears exactly once, as a DAG:
        assert set(report.rule_results) == {
            "adult", "resident", "kyc_complete", "high_income_debt_free"
        }
        assert report.rule_results["high_income_debt_free"].status \
            is RuleStatus.NOT_APPLICABLE

    def test_memoization_rule_evaluated_once(self):
        compiled = compile_document(MULTI)
        counter = {"n": 0}
        ev = Evaluator(coerce_facts(self.FACTS), compiled.rules_by_id)
        original = ev._comparison

        def counting(node, path):
            if node.fact == "age":
                counter["n"] += 1
            return original(node, path)

        ev._comparison = counting
        ev.report(MULTI)
        # 'adult' is referenced by kyc_complete AND directly by root,
        # but its age comparison runs exactly once.
        assert counter["n"] == 1

    def test_second_reference_points_at_shared_result(self):
        report = evaluate_document(MULTI, coerce_facts(self.FACTS))
        direct_adult_ref = report.root.children[1]
        assert direct_adult_ref.ref == "adult"
        assert direct_adult_ref.children == []  # no duplicated subtree
        assert direct_adult_ref.status \
            is report.rule_results["adult"].status

    def test_lazy_rules_never_reached_are_absent(self):
        facts = coerce_facts({"age": 15})  # kyc fails on 'adult'
        report = evaluate_document(MULTI, facts)
        assert report.decision is Decision.NOT_ELIGIBLE
        # Short-circuit: the gate rule was never reached, and the report
        # does not pretend otherwise.
        assert "high_income_debt_free" not in report.rule_results

    def test_unknown_propagates_through_references(self):
        facts = coerce_facts({"age": 34, "salary": 60_000, "debt_free": True})
        report = evaluate_document(MULTI, facts)  # residence unknown
        assert report.decision is Decision.NEEDS_REVIEW
        assert report.rule_results["kyc_complete"].status is RuleStatus.UNKNOWN


class TestApiCompileGate:
    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
        from app import storage, main
        importlib.reload(storage)
        importlib.reload(main)
        from fastapi.testclient import TestClient
        with TestClient(main.app) as c:
            yield c

    def test_publish_rejects_uncompilable_document(self, client):
        doc = {"name": "bad", "rules": [], "root": ref("nope")}
        r = client.post("/rulesets", json=doc)
        assert r.status_code == 422
        assert "unknown rule 'nope'" in r.json()["detail"]["compile_errors"][0]

    def test_publish_and_evaluate_multi_rule(self, client):
        r = client.post("/rulesets", json=MULTI.model_dump(mode="json"))
        assert r.status_code == 201
        r = client.post(
            "/rulesets/consumer-credit-de/evaluate",
            json={"facts": {"age": 34, "residence": "Germany",
                            "salary": 45_000}},
        )
        body = r.json()
        assert body["decision"] == "eligible"
        assert body["rule_results"]["kyc_complete"]["status"] == "satisfied"
