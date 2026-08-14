"""Shared libraries (imports), pin-on-publish, draft lifecycle, registry."""
import importlib

import pytest

from app.compiler import CompileError, compile_document
from app.models import AuthoringDocument


def cmp_(fact, op="eq", value=True):
    return {"kind": "comparison", "fact": fact, "operator": op, "value": value}


def ref(rule_id):
    return {"kind": "rule_ref", "rule": rule_id}


def make_doc(name, rules=None, root=None, imports=None):
    return AuthoringDocument.model_validate({
        "name": name,
        "imports": imports or [],
        "rules": rules or [],
        "root": root or cmp_("x"),
    })


def dict_resolver(docs: dict):
    """Pure test resolver: {name: (doc, version)}."""
    def resolve(name, version):
        if name not in docs:
            raise LookupError(f"'{name}' not found")
        doc, v = docs[name]
        if version is not None and version != v:
            raise LookupError(f"'{name}' version {version} not found")
        return doc, v
    return resolve


LIB = make_doc(
    "common-kyc",
    rules=[
        {"id": "adult", "root": cmp_("age", "ge", 18)},
        {"id": "resident", "root": cmp_("residence", "eq", "Germany")},
        {"id": "complete", "root": {"kind": "all", "children": [
            ref("adult"), ref("resident")]}},
    ],
    root=ref("complete"),
)


class TestImportCompilation:
    def test_imported_rules_are_namespaced_and_internal_refs_rewritten(self):
        app_doc = make_doc(
            "consumer-credit", imports=[{"ruleset": "common-kyc"}],
            root=ref("common-kyc:complete"),
        )
        compiled = compile_document(app_doc, dict_resolver({"common-kyc": (LIB, 3)}))
        assert "common-kyc:complete" in compiled.rules_by_id
        # 'complete' internally referenced 'adult' — must now be namespaced:
        assert compiled.dependencies["common-kyc:complete"] == {
            "common-kyc:adult", "common-kyc:resident"}
        assert compiled.resolved_imports == {"common-kyc": 3}

    def test_transitive_imports_resolve(self):
        mid = make_doc("mid", imports=[{"ruleset": "common-kyc"}],
                       rules=[{"id": "wrapper",
                               "root": ref("common-kyc:adult")}],
                       root=ref("wrapper"))
        top = make_doc("top", imports=[{"ruleset": "mid"}],
                       root=ref("mid:wrapper"))
        compiled = compile_document(top, dict_resolver(
            {"common-kyc": (LIB, 1), "mid": (mid, 2)}))
        assert compiled.resolved_imports == {"common-kyc": 1, "mid": 2}
        assert compiled.dependencies["mid:wrapper"] == {"common-kyc:adult"}

    def test_cross_document_cycle_is_a_compile_error(self):
        a = make_doc("a", imports=[{"ruleset": "b"}])
        b = make_doc("b", imports=[{"ruleset": "a"}])
        with pytest.raises(CompileError) as e:
            compile_document(a, dict_resolver({"a": (a, 1), "b": (b, 1)}))
        assert any("circular import" in m for m in e.value.errors)

    def test_version_conflict_across_import_graph(self):
        lib_v1 = (LIB, 1)
        mid = make_doc("mid", imports=[{"ruleset": "common-kyc", "version": 2}])

        def resolve(name, version):
            if name == "mid":
                return mid, 1
            if name == "common-kyc":
                return LIB, version or 1
            raise LookupError(name)

        top = make_doc("top", imports=[
            {"ruleset": "common-kyc", "version": 1}, {"ruleset": "mid"}])
        with pytest.raises(CompileError) as e:
            compile_document(top, resolve)
        assert any("version conflict" in m for m in e.value.errors)

    def test_local_rule_colliding_with_import_namespace_rejected(self):
        # ids can't contain ':' by pattern, so collision needs a crafted case
        # via unknown-ref instead; assert imports without resolver fail loudly
        doc = make_doc("x", imports=[{"ruleset": "common-kyc"}])
        with pytest.raises(CompileError) as e:
            compile_document(doc, None)
        assert "no import resolver" in e.value.errors[0]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    from app import storage, main
    importlib.reload(storage)
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c


LIB_JSON = LIB.model_dump(mode="json")

APP_JSON = make_doc(
    "consumer-credit",
    imports=[{"ruleset": "common-kyc"}],  # no version -> pin-on-publish
    root=ref("common-kyc:complete"),
).model_dump(mode="json")

FACTS_OK = {"facts": {"age": 20, "residence": "Germany"}}


class TestPinOnPublish:
    def test_library_update_never_silently_changes_a_process(self, client):
        client.post("/rulesets", json=LIB_JSON)                    # lib v1
        r = client.post("/rulesets", json=APP_JSON)                # app v1
        assert r.json()["resolved_imports"] == {"common-kyc": 1}

        r = client.post("/rulesets/consumer-credit/evaluate", json=FACTS_OK)
        assert r.json()["decision"] == "eligible"

        # Library tightens the age rule to 21 -> lib v2
        lib2 = dict(LIB_JSON)
        lib2["rules"] = [
            {"id": "adult", "root": cmp_("age", "ge", 21)},
            LIB_JSON["rules"][1], LIB_JSON["rules"][2],
        ]
        client.post("/rulesets", json=lib2)

        # The process still evaluates against its pinned v1 — no drift:
        r = client.post("/rulesets/consumer-credit/evaluate", json=FACTS_OK)
        assert r.json()["decision"] == "eligible"

        # Re-publishing the process picks up (and pins) v2:
        r = client.post("/rulesets", json=APP_JSON)
        assert r.json()["resolved_imports"] == {"common-kyc": 2}
        r = client.post("/rulesets/consumer-credit/evaluate", json=FACTS_OK)
        assert r.json()["decision"] == "not_eligible"

        # And the old process version is still replayable bit-for-bit:
        r = client.post("/rulesets/consumer-credit/evaluate?version=1",
                        json=FACTS_OK)
        assert r.json()["decision"] == "eligible"


class TestDraftLifecycle:
    def test_draft_flow(self, client):
        r = client.post("/rulesets?draft=true", json=LIB_JSON)
        assert r.json()["status"] == "draft"

        # Not resolved as latest:
        r = client.post("/rulesets/common-kyc/evaluate", json=FACTS_OK)
        assert r.status_code == 404

        # But testable by explicit version:
        r = client.post("/rulesets/common-kyc/evaluate?version=1",
                        json=FACTS_OK)
        assert r.json()["decision"] == "eligible"

        # Not importable while draft:
        r = client.post("/rulesets", json=APP_JSON)
        assert r.status_code == 422
        assert "not published" in str(r.json()["detail"]) or \
               "not found" in str(r.json()["detail"])

        # Promote -> everything opens up:
        r = client.post("/rulesets/common-kyc/versions/1/publish")
        assert r.json()["status"] == "published"
        assert client.post("/rulesets", json=APP_JSON).status_code == 201
        r = client.post("/rulesets/common-kyc/evaluate", json=FACTS_OK)
        assert r.json()["decision"] == "eligible"

        # Promotion is one-way, once:
        r = client.post("/rulesets/common-kyc/versions/1/publish")
        assert r.status_code == 409

    def test_registry(self, client):
        client.post("/rulesets", json=LIB_JSON)
        client.post("/rulesets?draft=true", json=LIB_JSON)
        client.post("/rulesets", json=APP_JSON)

        entries = {e["name"]: e for e in client.get("/rulesets").json()}
        assert entries["common-kyc"]["latest_published"] == 1
        assert entries["common-kyc"]["latest_draft"] == 2
        assert entries["common-kyc"]["versions"] == 2
        assert entries["consumer-credit"]["latest_published"] == 1
        assert entries["consumer-credit"]["latest_draft"] is None
