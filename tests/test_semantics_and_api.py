"""Group semantics from the deck + fact statuses + full API round-trip."""
import pytest

from app.evaluator import Evaluator, evaluate_document
from app.models import AuthoringDocument, coerce_facts
from app.report import RuleStatus


def doc(root: dict) -> AuthoringDocument:
    return AuthoringDocument.model_validate({"name": "t", "root": root})


def cmp_(fact, value=True):
    return {"kind": "comparison", "fact": fact, "operator": "eq", "value": value}


GATE_A = {  # WHEN gate_a REQUIRE req_a  -> not_applicable when gate_a false
    "kind": "conditional_requirement",
    "when": cmp_("gate_a"), "require": cmp_("req_a"),
}


def status_of(root, facts, **kw) -> RuleStatus:
    ev = Evaluator(coerce_facts(facts), **kw)
    return ev.evaluate(doc(root)).status


class TestAllSemantics:
    def test_satisfied_plus_not_applicable_is_satisfied(self):
        root = {"kind": "all", "children": [cmp_("a"), cmp_("b"), GATE_A]}
        facts = {"a": True, "b": True, "gate_a": False}
        assert status_of(root, facts) is RuleStatus.SATISFIED

    def test_failed_beats_unknown(self):
        root = {"kind": "all", "children": [cmp_("a"), cmp_("b")]}
        facts = {"a": False}  # b unknown
        assert status_of(root, facts) is RuleStatus.FAILED

    def test_unknown_beats_not_applicable(self):
        root = {"kind": "all", "children": [GATE_A, cmp_("b")]}
        facts = {"gate_a": False}  # b unknown
        assert status_of(root, facts) is RuleStatus.UNKNOWN

    def test_all_not_applicable(self):
        root = {"kind": "all", "children": [GATE_A]}
        assert status_of(root, {"gate_a": False}) is RuleStatus.NOT_APPLICABLE


class TestAnySemantics:
    def test_satisfied_wins(self):
        root = {"kind": "any", "children": [cmp_("a"), cmp_("b")]}
        assert status_of(root, {"a": True}) is RuleStatus.SATISFIED

    def test_unknown_before_failed(self):
        root = {"kind": "any", "children": [cmp_("a"), cmp_("b")]}
        assert status_of(root, {"a": False}) is RuleStatus.UNKNOWN

    def test_every_child_na(self):
        root = {"kind": "any", "children": [GATE_A]}
        assert status_of(root, {"gate_a": False}) is RuleStatus.NOT_APPLICABLE

    def test_otherwise_failed(self):
        root = {"kind": "any", "children": [cmp_("a"), cmp_("b")]}
        assert status_of(root, {"a": False, "b": False}) is RuleStatus.FAILED


class TestNotAndFacts:
    def test_not_flips_and_passes_through(self):
        root = {"kind": "not", "child": cmp_("a")}
        assert status_of(root, {"a": True}) is RuleStatus.FAILED
        assert status_of(root, {"a": False}) is RuleStatus.SATISFIED
        assert status_of(root, {}) is RuleStatus.UNKNOWN

    def test_na_input_policy_default_unknown(self):
        root = cmp_("a")
        facts = {"a": {"status": "not_applicable"}}
        assert status_of(root, facts) is RuleStatus.UNKNOWN
        assert status_of(root, facts, na_input_policy="not_applicable") \
            is RuleStatus.NOT_APPLICABLE

    def test_error_status_kept_apart_from_unknown(self):
        root = cmp_("a")
        assert status_of(root, {"a": {"status": "error"}}) is RuleStatus.ERROR

    def test_type_mismatch_is_error_not_failed(self):
        root = {"kind": "comparison", "fact": "age", "operator": "ge",
                "value": 28}
        assert status_of(root, {"age": "thirty"}) is RuleStatus.ERROR


class TestBetween:
    ROOT = {"kind": "comparison", "fact": "age", "operator": "between",
            "value": [18, 65]}

    def test_inclusive_boundaries(self):
        for age in (18, 34, 65):
            assert status_of(self.ROOT, {"age": age}) is RuleStatus.SATISFIED
        for age in (17, 66):
            assert status_of(self.ROOT, {"age": age}) is RuleStatus.FAILED

    def test_unknown_fact_is_unknown(self):
        assert status_of(self.ROOT, {}) is RuleStatus.UNKNOWN

    def test_type_mismatch_is_error(self):
        assert status_of(self.ROOT, {"age": "thirty"}) is RuleStatus.ERROR

    @pytest.mark.parametrize("value", [18, [18], [1, 2, 3], "18-65"])
    def test_value_must_be_two_item_list(self, value):
        with pytest.raises(ValueError, match="two-item list"):
            doc({"kind": "comparison", "fact": "age",
                 "operator": "between", "value": value})


class TestApi:
    @pytest.fixture()
    def client(self, tmp_path, monkeypatch):
        import importlib
        monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
        from app import storage, main
        importlib.reload(storage)
        importlib.reload(main)
        from fastapi.testclient import TestClient
        with TestClient(main.app) as c:
            yield c

    def test_publish_versioning_and_evaluate(self, client):
        document = {
            "name": "demo",
            "root": {"kind": "comparison", "fact": "age",
                     "operator": "ge", "value": 28},
        }
        r1 = client.post("/rulesets", json=document)
        assert r1.status_code == 201 and r1.json()["version"] == 1

        document["root"]["value"] = 21  # policy change -> new version
        r2 = client.post("/rulesets", json=document)
        assert r2.json()["version"] == 2

        # latest version applies by default
        r = client.post("/rulesets/demo/evaluate", json={"facts": {"age": 25}})
        assert r.json()["decision"] == "eligible"

        # pinning the old version replays the old policy
        r = client.post("/rulesets/demo/evaluate?version=1",
                        json={"facts": {"age": 25}})
        assert r.json()["decision"] == "not_eligible"

        # unknown facts surface as needs_review, never guessed as false
        r = client.post("/rulesets/demo/evaluate", json={"facts": {}})
        assert r.json()["decision"] == "needs_review"

    def test_invalid_document_rejected_at_the_edge(self, client):
        bad = {"name": "demo", "root": {"kind": "comparison"}}  # missing fields
        r = client.post("/rulesets", json=bad)
        assert r.status_code == 422
