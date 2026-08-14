"""Block builder: raw-dict state round-trips, structural actions, publish."""
import html
import importlib
import json
import re

import pytest


def cmp_(fact, op="eq", value=True):
    return {"kind": "comparison", "fact": fact, "operator": op, "value": value}


def ref(rule_id):
    return {"kind": "rule_ref", "rule": rule_id}


LIB = {
    "name": "common-kyc",
    "rules": [
        {"id": "adult", "root": cmp_("age", "ge", 18)},
        {"id": "complete", "root": {"kind": "all", "children": [ref("adult")]}},
    ],
    "root": ref("complete"),
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
        yield c


def state(resp) -> dict:
    """Extract the builder's hidden doc_json state back out of the page."""
    m = re.search(r"name='doc_json' value='([^']*)'", resp.text)
    assert m, "no doc_json field in page"
    return json.loads(html.unescape(m.group(1)))


def post(client, doc, action, follow_redirects=True, **fields):
    data = {"doc_json": json.dumps(doc), "action": action, **fields}
    return client.post("/ui/build", data=data,
                       follow_redirects=follow_redirects)


BASE = {"name": "built", "root": cmp_("age", "ge", 18)}


class TestPages:
    def test_bare_get_renders_starter(self, client):
        r = client.get("/ui/build")
        assert r.status_code == 200
        assert "Rule set builder" in r.text
        assert state(r)["name"] == "my-ruleset"
        assert ">between</option>" in r.text  # new operator in the select

    def test_from_prefills_existing_version(self, client):
        r = client.get("/ui/build?from=common-kyc&version=1")
        assert state(r)["rules"][0]["id"] == "adult"

    def test_from_unknown_404(self, client):
        assert client.get("/ui/build?from=nope").status_code == 404

    def test_first_submit_button_is_hidden_noop(self, client):
        r = client.get("/ui/build")
        first = r.text.index("<button")
        assert "value='' hidden" in r.text[first:first + 80]

    def test_document_page_links_builder(self, client):
        assert "Edit in builder" in client.get("/ui/common-kyc").text


class TestActions:
    def test_set_root_kind(self, client):
        s = state(post(client, BASE, "set:root", **{"k:root": "all"}))
        assert s["root"]["kind"] == "all"
        assert s["root"]["children"][0]["kind"] == "comparison"

    def test_add_child(self, client):
        doc = dict(BASE, root={"kind": "all", "children": [cmp_("a")]})
        s = state(post(client, doc, "add:root", **{"k:root": "not"}))
        assert len(s["root"]["children"]) == 2
        assert s["root"]["children"][1]["kind"] == "not"

    def test_delete_child(self, client):
        doc = dict(BASE, root={"kind": "all",
                               "children": [cmp_("a"), cmp_("b")]})
        s = state(post(client, doc, "del:root.children[0]"))
        assert [c["fact"] for c in s["root"]["children"]] == ["b"]

    def test_empty_group_allowed_midedit_rejected_on_validate(self, client):
        doc = dict(BASE, root={"kind": "all", "children": [cmp_("a")]})
        s = state(post(client, doc, "del:root.children[0]"))
        assert s["root"]["children"] == []
        r = post(client, s, "validate")
        assert "Invalid document" in r.text
        assert state(r)["root"]["children"] == []  # state preserved

    def test_addrule_generates_fresh_ids(self, client):
        s = state(post(client, BASE, "addrule"))
        assert s["rules"][0]["id"] == "rule-1"
        s = state(post(client, s, "addrule"))
        assert [r["id"] for r in s["rules"]] == ["rule-1", "rule-2"]

    def test_delrule(self, client):
        s = state(post(client, dict(BASE, rules=[
            {"id": "a", "root": cmp_("x")}]), "delrule:0"))
        assert s["rules"] == []

    def test_add_and_del_import(self, client):
        s = state(post(client, BASE, "addimport"))
        assert s["imports"] == [{"ruleset": "", "version": None}]
        s = state(post(client, s, "delimport:0"))
        assert s["imports"] == []

    def test_empty_action_applies_overlay(self, client):
        s = state(post(client, BASE, "",
                       **{"d:name": "renamed", "n:root:fact": "income"}))
        assert s["name"] == "renamed"
        assert s["root"]["fact"] == "income"

    def test_tampered_path_shows_error_not_500(self, client):
        r = post(client, BASE, "del:root")
        assert r.status_code == 200
        assert "could not apply edit" in r.text


class TestScalarOverlay:
    def test_between_value_round_trips_as_list(self, client):
        s = state(post(client, BASE, "", **{
            "n:root:operator": "between", "n:root:value": "[18, 65]"}))
        assert s["root"]["operator"] == "between"
        assert s["root"]["value"] == [18, 65]

    def test_bare_string_value(self, client):
        s = state(post(client, BASE, "", **{"n:root:value": "Germany"}))
        assert s["root"]["value"] == "Germany"

    def test_rule_attrs_and_import_version(self, client):
        doc = dict(BASE, rules=[{"id": "a", "root": cmp_("x")}],
                   imports=[{"ruleset": "common-kyc", "version": None}])
        s = state(post(client, doc, "", **{
            "r:0:id": "adult-check", "r:0:label": "Adult",
            "r:0:automation": "manual", "i:0:version": "1"}))
        assert s["rules"][0] == {"id": "adult-check", "label": "Adult",
                                 "automation": "manual", "root": cmp_("x")}
        assert s["imports"][0]["version"] == 1


class TestPublish:
    def test_build_between_publish_and_evaluate(self, client):
        doc = {"name": "span-credit", "root": cmp_("age")}
        s = state(post(client, doc, "", **{
            "n:root:operator": "between", "n:root:value": "[18, 65]"}))
        r = post(client, s, "publish", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/ui/span-credit?version=1"

        stored = client.get("/rulesets/span-credit/versions/1").json()
        assert stored["root"]["operator"] == "between"
        assert stored["root"]["value"] == [18, 65]

        page = client.get("/ui/span-credit")
        assert "between" in page.text

        ev = client.post("/ui/span-credit/evaluate?version=1",
                         data={"f:age": "65", "s:age": "known"})
        assert "Not eligible" not in ev.text
        assert "Eligible" in ev.text  # boundary is inclusive

    def test_publish_draft_not_latest(self, client):
        doc = {"name": "built-draft", "root": cmp_("x")}
        post(client, doc, "publish", draft="1", follow_redirects=False)
        assert client.get("/ui/built-draft").status_code == 404
        assert client.get("/ui/built-draft?version=1").status_code == 200

    def test_publish_pins_imports(self, client):
        doc = {"name": "built-pins",
               "imports": [{"ruleset": "common-kyc", "version": None}],
               "root": ref("common-kyc:complete")}
        post(client, doc, "publish")
        stored = client.get("/rulesets/built-pins/versions/1").json()
        assert stored["imports"][0]["version"] == 1

    def test_validate_previews_without_persisting(self, client):
        r = post(client, BASE, "validate")
        assert "Preview" in r.text
        assert client.get("/rulesets/built/versions").status_code == 404


class TestErrors:
    def test_dangling_ref_compile_error_preserves_state(self, client):
        doc = {"name": "dangling", "root": ref("missing")}
        r = post(client, doc, "publish")
        assert "Compile errors" in r.text
        assert state(r)["root"]["rule"] == "missing"
        assert client.get("/rulesets/dangling/versions").status_code == 404

    def test_duplicate_rule_ids_surface_on_validate(self, client):
        doc = dict(BASE, rules=[{"id": "a", "root": cmp_("x")},
                                {"id": "a", "root": cmp_("y")}])
        r = post(client, doc, "validate")
        assert "Invalid document" in r.text

    def test_malformed_state_rescued_in_json_editor(self, client):
        r = client.post("/ui/build", data={"doc_json": "{nope", "action": ""})
        assert r.status_code == 200
        assert "builder state unreadable" in r.text
        assert "{nope" in r.text  # nothing lost


class TestHandoffs:
    def test_tojson_opens_json_editor_with_doc(self, client):
        r = post(client, BASE, "tojson")
        assert "/ui/new" in r.text
        assert "&quot;built&quot;" in r.text

    def test_tobuilder_from_json_editor(self, client):
        r = client.post("/ui/new", data={
            "doc_json": json.dumps(BASE), "action": "tobuilder"})
        assert "Rule set builder" in r.text
        assert state(r)["name"] == "built"

    def test_tobuilder_with_uncompilable_doc_still_works(self, client):
        doc = {"name": "wip", "root": ref("missing")}  # dangling ref is fine
        r = client.post("/ui/new", data={
            "doc_json": json.dumps(doc), "action": "tobuilder"})
        assert "Rule set builder" in r.text
