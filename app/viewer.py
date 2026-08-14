"""Read-only web viewer — server-rendered HTML straight from FastAPI.

Design notes:

* The AuthoringDocument is already UI-shaped (nested, labeled, small
  vocabulary), so the viewer walks the same tree the evaluator walks and
  generates the *same node paths*. Evaluation results overlay purely by path
  lookup — the only coupling between viewer and engine is the stable path
  convention that exists since v0.1.
* Zero frontend build: no npm, no framework, one embedded stylesheet.
  This is deliberately the reviewer-facing rendering, not the authoring UI.
* Not part of the JSON API surface (``include_in_schema=False``).
"""
from __future__ import annotations

import difflib
import html
import json

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from . import storage
from .compiler import CompiledDocument, CompileError, compile_document
from .evaluator import _OPS, Evaluator
from .models import (
    AllBlock,
    AnyBlock,
    AuthoringDocument,
    Comparison,
    ConditionalRequirement,
    FactInput,
    FactStatus,
    ImportSpec,
    NotBlock,
    OneOfBlock,
    RuleRef,
    coerce_facts,
)
from .report import STATUS_TO_DECISION, EvaluationReport, RuleResult
from .statements import DEFAULT_CONFIDENCE_THRESHOLD, Statement, resolve_facts

router = APIRouter(prefix="/ui", include_in_schema=False)


def esc(x) -> str:
    return html.escape(str(x))


_CSS = """
:root{--ink:#1f2430;--mut:#6b7280;--line:#e5e7eb;--card:#fff;--bg:#f7f7fa;
--accent:#6d28d9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:20px 14px 60px}
header{margin-bottom:18px}.brand{color:var(--accent);font-weight:600;
text-decoration:none}
h1{font-size:22px;margin:6px 0 2px}h2{font-size:16px;margin:26px 0 10px}
.mut{color:var(--mut);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:14px;margin:10px 0}
.node{border:1px solid var(--line);border-radius:8px;padding:9px 11px;
margin:8px 0;background:var(--card)}
.node .kids{margin-left:10px;border-left:2px solid var(--line);
padding-left:10px}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px;
flex-wrap:wrap}
.op{color:var(--accent)}.lbl{font-weight:600;margin-bottom:2px}
.reason{color:var(--mut);font-size:12.5px;margin-top:4px}
.hd{font-weight:600;color:var(--accent);font-size:13px;
text-transform:uppercase;letter-spacing:.4px;margin-bottom:4px}
.chip{font-size:12px;padding:2px 10px;border-radius:999px;white-space:nowrap}
.s-satisfied{color:#0a7d33;background:#e6f4ea}
.s-failed{color:#b3261e;background:#fdecea}
.s-unknown{color:#9a6700;background:#fff3d6}
.s-not_applicable{color:#6d28d9;background:#f1e9fe}
.s-error{color:#7f1d1d;background:#fee2e2;border:1px solid #fecaca}
.s-not_evaluated{color:#6b7280;background:#f3f4f6}
.banner{display:flex;justify-content:space-between;align-items:center;
padding:12px 14px;border-radius:10px;border:1px solid var(--line);
background:var(--card);margin:12px 0;font-weight:600}
.banner .chip{font-size:14px;padding:4px 14px}
table{border-collapse:collapse;width:100%;background:var(--card)}
td,th{border:1px solid var(--line);padding:8px 10px;text-align:left;
font-size:14px}th{background:#faf5ff}
textarea{width:100%;font:13px/1.5 ui-monospace,Consolas,monospace;
border:1px solid var(--line);border-radius:8px;padding:10px;min-height:150px}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;
padding:9px 18px;font-size:14px;margin-top:8px;cursor:pointer}
a{color:var(--accent)}
.err{background:#fdecea;border:1px solid #f5c6c3;color:#b3261e;
border-radius:8px;padding:10px 12px;margin:10px 0}
.vtag{font-size:12px;background:#f1e9fe;color:#6d28d9;border-radius:6px;
padding:1px 8px;margin-left:6px}
input[type=text],input[type=number],input:not([type]),select{
font:13px/1.5 ui-monospace,Consolas,monospace;border:1px solid var(--line);
border-radius:6px;padding:5px 8px;background:var(--card)}
details{margin:8px 0}details summary{cursor:pointer;color:var(--mut);
font-size:13px}details textarea{min-height:90px;margin-top:6px}
.actions{display:flex;gap:12px;align-items:center;flex-wrap:wrap;
font-size:13px;margin-top:6px}
.actions button{margin-top:0;padding:5px 12px;font-size:13px}
pre.diff{background:var(--card);border:1px solid var(--line);
border-radius:10px;padding:12px;overflow-x:auto;
font:12.5px/1.5 ui-monospace,Consolas,monospace}
.dadd{background:#e6f4ea}.ddel{background:#fdecea}
"""

_DECISION_LABEL = {
    "eligible": ("Eligible", "s-satisfied"),
    "not_eligible": ("Not eligible", "s-failed"),
    "needs_review": ("Needs review", "s-unknown"),
    "not_applicable": ("Not applicable", "s-not_applicable"),
    "error": ("Error", "s-error"),
}


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(title)}</title><style>{_CSS}</style></head><body>"
        "<div class='wrap'><header><a class='brand' href='/ui'>"
        "Eligibility Rule Engine</a></header>"
        f"{body}</div></body></html>"
    )


# ---------------------------------------------------------------------------
# Result index: path -> RuleResult, built once per report
# ---------------------------------------------------------------------------

def build_result_index(report: EvaluationReport) -> dict[str, RuleResult]:
    index: dict[str, RuleResult] = {}

    def walk(r: RuleResult) -> None:
        index[r.path] = r
        for c in r.children:
            walk(c)

    walk(report.root)
    for r in report.rule_results.values():
        walk(r)
    return index


def _chip(result: RuleResult | None, evaluated: bool) -> str:
    if not evaluated:
        return ""
    if result is None:  # lazily skipped named rule — honestly not evaluated
        return "<span class='chip s-not_evaluated'>not evaluated</span>"
    s = result.status.value
    return f"<span class='chip s-{s}'>{esc(s.replace('_', ' '))}</span>"


# ---------------------------------------------------------------------------
# Document rendering — walks the tree with the evaluator's path convention
# ---------------------------------------------------------------------------

def _render_node(node, path: str, index: dict | None) -> str:
    evaluated = index is not None
    result = index.get(path) if evaluated else None
    chip = _chip(result, evaluated)
    label = f"<div class='lbl'>{esc(node.label)}</div>" if node.label else ""
    reason = (
        f"<div class='reason'>{esc(result.reason)}</div>"
        if result is not None else ""
    )

    if isinstance(node, Comparison):
        sym = _OPS[node.operator][0]
        text = (f"<b>{esc(node.fact)}</b> <span class='op'>{esc(sym)}</span> "
                f"{esc(json.dumps(node.value))}")
        return (f"<div class='node'>{label}<div class='row'>"
                f"<span>{text}</span>{chip}</div>{reason}</div>")

    if isinstance(node, (AllBlock, AnyBlock, OneOfBlock)):
        head = ("ALL of the following" if isinstance(node, AllBlock)
                else "EXACTLY ONE of the following"
                if isinstance(node, OneOfBlock)
                else "AT LEAST ONE of the following")
        kids = "".join(
            _render_node(c, f"{path}.children[{i}]", index)
            for i, c in enumerate(node.children)
        )
        return (f"<div class='node'>{label}<div class='row'>"
                f"<span class='hd'>{head}</span>{chip}</div>"
                f"<div class='kids'>{kids}</div></div>")

    if isinstance(node, NotBlock):
        kid = _render_node(node.child, f"{path}.child", index)
        return (f"<div class='node'>{label}<div class='row'>"
                f"<span class='hd'>It is NOT the case that</span>{chip}</div>"
                f"<div class='kids'>{kid}</div></div>")

    if isinstance(node, ConditionalRequirement):
        when = _render_node(node.when, f"{path}.when", index)
        req = _render_node(node.require, f"{path}.require", index)
        return (f"<div class='node'>{label}<div class='row'>"
                f"<span class='hd'>Conditional requirement</span>{chip}</div>"
                f"{reason}<div class='kids'>"
                f"<div class='mut'>Apply only when:</div>{when}"
                f"<div class='mut'>Then require:</div>{req}</div></div>")

    if isinstance(node, RuleRef):
        anchor = esc(node.rule).replace(":", "--")
        return (f"<div class='node'>{label}<div class='row'><span>Rule: "
                f"<a href='#rule-{anchor}'>{esc(node.label or node.rule)}</a>"
                f"</span>{chip}</div>{reason}</div>")

    return f"<div class='node'>unsupported block: {esc(node.kind)}</div>"


def collect_fact_names(compiled: CompiledDocument) -> list[str]:
    names: set[str] = set()

    def walk(node) -> None:
        if isinstance(node, Comparison):
            names.add(node.fact)
        elif isinstance(node, (AllBlock, AnyBlock, OneOfBlock)):
            for c in node.children:
                walk(c)
        elif isinstance(node, NotBlock):
            walk(node.child)
        elif isinstance(node, ConditionalRequirement):
            walk(node.when)
            walk(node.require)

    walk(compiled.doc.root)
    for rule in compiled.rules_by_id.values():
        walk(rule.root)
    return sorted(names)


def _rules_cards(compiled: CompiledDocument, index: dict | None) -> str:
    if not compiled.rules_by_id:
        return ""
    parts = ["<h2>Named rules</h2>"]
    for rid in sorted(compiled.rules_by_id):
        rule = compiled.rules_by_id[rid]
        anchor = esc(rid).replace(":", "--")
        head = esc(rule.label or rid)
        sub = (f"<span class='mut'> — {esc(rid)}</span>"
               if rule.label else "")
        tree = _render_node(rule.root, f"rule:{rid}", index)
        result = index.get(f"rule:{rid}") if index else None
        chip = _chip(result, index is not None)
        parts.append(
            f"<div class='card' id='rule-{anchor}'>"
            f"<div class='row'><span class='lbl'>{head}{sub}</span>"
            f"{chip}</div>{tree}</div>"
        )
    return "".join(parts)


def _facts_form(
    compiled: CompiledDocument,
    row,
    report: EvaluationReport | None,
    facts_json: str | None,
    statements_json: str | None,
    threshold: float,
) -> str:
    doc = compiled.doc
    report_facts = report.facts if report else {}
    fact_rows = []
    for name in collect_fact_names(compiled):
        fi = report_facts.get(name)
        val = ("" if fi is None or fi.value is None
               else esc(json.dumps(fi.value)))
        status = fi.status.value if fi is not None else "known"
        opts = "".join(
            f"<option value='{s}'{' selected' if s == status else ''}>"
            f"{s.replace('_', ' ')}</option>"
            for s in ("known", "unknown", "not_applicable")
        )
        fact_rows.append(
            f"<tr><td>{esc(name)}</td>"
            f"<td><input name='f:{esc(name)}' value='{val}'></td>"
            f"<td><select name='s:{esc(name)}'>{opts}</select></td></tr>"
        )

    if facts_json is None:
        skeleton = {name: None for name in collect_fact_names(compiled)}
        facts_json = json.dumps(skeleton, indent=1)
    if statements_json is None:
        statements_json = "[]"

    return (
        f"<div class='card'><div class='hd'>Facts</div>"
        f"<form method='post' "
        f"action='/ui/{esc(doc.name)}/evaluate?version={row.version}'>"
        f"<table><tr><th>Fact</th><th>Value</th><th>Status</th></tr>"
        f"{''.join(fact_rows)}</table>"
        f"<div class='mut'>values are parsed as JSON, bare text is fine; "
        f"empty + known = not provided → unknown</div>"
        f"<details><summary>Raw JSON / statements (fact rows win over raw "
        f"facts, raw facts win over statements)</summary>"
        f"<div class='mut'>facts JSON — null = not provided</div>"
        f"<textarea name='facts_json'>{esc(facts_json)}</textarea>"
        f"<div class='mut'>statements JSON — resolved to facts "
        f"(override &gt; confidence &gt; recency)</div>"
        f"<textarea name='statements_json'>{esc(statements_json)}</textarea>"
        f"<label class='mut'>confidence threshold "
        f"<input type='number' name='confidence_threshold' step='0.05' "
        f"min='0' max='1' value='{threshold}'></label>"
        f"</details>"
        f"<button type='submit'>Evaluate v{row.version}</button></form></div>"
    )


def _document_page(
    row,
    compiled: CompiledDocument,
    all_versions,
    report: EvaluationReport | None = None,
    facts_json: str | None = None,
    statements_json: str | None = None,
    threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    error: str | None = None,
    note: str | None = None,
) -> str:
    doc = compiled.doc
    index = build_result_index(report) if report else None
    parts: list[str] = []

    vtags = " ".join(
        (f"<b>v{v.version}</b>" if v.version == row.version else
         f"<a href='/ui/{esc(doc.name)}?version={v.version}'>v{v.version}</a>")
        + f"<span class='vtag'>{esc(v.status)}</span>"
        for v in all_versions
    )
    parts.append(f"<h1>{esc(doc.name)}</h1><div class='mut'>{vtags}</div>")

    actions = [
        f"<a href='/ui/new?from={esc(doc.name)}&version={row.version}'>"
        f"Edit as new version</a>",
        f"<a href='/ui/{esc(doc.name)}/history'>History</a>",
    ]
    prior = [v.version for v in all_versions if v.version < row.version]
    if prior:
        p = max(prior)
        actions.append(
            f"<a href='/ui/{esc(doc.name)}/diff?a={p}&b={row.version}'>"
            f"diff vs v{p}</a>")
    promote = ""
    if row.status == "draft":
        promote = (
            f"<form method='post' action='/ui/{esc(doc.name)}/promote"
            f"?version={row.version}'>"
            f"<button type='submit'>Promote to published</button></form>")
    parts.append(f"<div class='actions'>{' '.join(actions)}{promote}</div>")

    if doc.description:
        parts.append(f"<p class='mut'>{esc(doc.description)}</p>")

    if error:
        parts.append(f"<div class='err'>{esc(error)}</div>")
    if note:
        parts.append(f"<p class='mut'>{esc(note)}</p>")

    if report:
        text, cls = _DECISION_LABEL[report.decision.value]
        parts.append(
            f"<div class='banner'><span>Decision</span>"
            f"<span class='chip {cls}'>{esc(text)}</span></div>"
        )

    # Facts form — one row per fact the rules read, prefilled from the
    # report after an evaluation; raw JSON / statements in <details>.
    parts.append(_facts_form(compiled, row, report, facts_json,
                             statements_json, threshold))

    if doc.imports:
        rows = "".join(
            f"<tr><td><a href='/ui/{esc(s.ruleset)}?version={s.version}'>"
            f"{esc(s.ruleset)}</a></td><td>v{s.version}</td></tr>"
            for s in doc.imports
        )
        parts.append(f"<h2>Imports (pinned)</h2><table>"
                     f"<tr><th>Library</th><th>Version</th></tr>{rows}</table>")

    parts.append("<h2>Decision</h2>")
    parts.append(_render_node(doc.root, "root", index))
    parts.append(_rules_cards(compiled, index))

    return _page(doc.name, "".join(parts))


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def get_session():
    with storage.SessionLocal() as session:
        with session.begin():
            yield session


def _load(session: Session, name: str, version: int | None):
    row = storage.get_version(session, name, version)
    if row is None:
        raise HTTPException(404, f"rule set '{name}' not found "
                            "(or has no published version)")
    doc = AuthoringDocument.model_validate(row.document)
    compiled = compile_document(doc, storage.import_resolver(session))
    return row, compiled


@router.get("", response_class=HTMLResponse)
def registry_page(session: Session = Depends(get_session)) -> str:
    entries = storage.registry(session)
    if not entries:
        body = "<h1>Rule sets</h1><p class='mut'>Nothing published yet.</p>"
        return _page("Rule sets", body)
    rows = "".join(
        f"<tr><td><a href='/ui/{esc(e['name'])}'>{esc(e['name'])}</a></td>"
        f"<td>{e['latest_published'] or '—'}</td>"
        f"<td>{e['latest_draft'] or '—'}</td>"
        f"<td>{e['versions']}</td></tr>"
        for e in entries
    )
    body = (
        "<h1>Rule sets</h1>"
        "<table><tr><th>Name</th><th>Latest published</th>"
        f"<th>Latest draft</th><th>Versions</th></tr>{rows}</table>"
    )
    return _page("Rule sets", body)


# Authoring. Registered before /{name} so it isn't shadowed — side effect:
# a rule set literally named 'new' is unreachable in the UI. POC ceiling.

_STARTER_DOC = {
    "name": "my-ruleset",
    "description": "Describe the decision this rule set makes.",
    "root": {"kind": "comparison", "fact": "age", "operator": "ge",
             "value": 18},
}


def _authoring_page(
    doc_json: str,
    error: str | None = None,
    errors: list[str] | None = None,
    preview: str | None = None,
) -> str:
    parts = [
        "<h1>New rule set version</h1>",
        "<p class='mut'>Publishing always creates a new immutable version "
        "(append-only). Validate previews without saving.</p>",
    ]
    if error:
        parts.append(f"<div class='err'>{esc(error)}</div>")
    if errors:
        items = "".join(f"<li>{esc(e)}</li>" for e in errors)
        parts.append(f"<div class='err'>Compile errors:<ul>{items}</ul></div>")
    parts.append(
        f"<form method='post' action='/ui/new'>"
        f"<textarea name='doc_json' style='min-height:320px'>"
        f"{esc(doc_json)}</textarea>"
        f"<div class='actions'>"
        f"<button type='submit' name='action' value='validate'>"
        f"Validate &amp; preview</button>"
        f"<button type='submit' name='action' value='publish'>"
        f"Publish</button>"
        f"<label class='mut'><input type='checkbox' name='draft' value='1'> "
        f"publish as draft</label>"
        f"</div></form>"
    )
    if preview:
        parts.append("<h2>Preview</h2>" + preview)
    return _page("New rule set", "".join(parts))


@router.get("/new", response_class=HTMLResponse)
def authoring_page(
    from_: str | None = Query(default=None, alias="from"),
    version: int | None = Query(default=None),
    session: Session = Depends(get_session),
) -> str:
    """Create a rule set — or, with ?from=<name>&version=<n>, edit an
    existing version by prefilling it (publishing creates a new version)."""
    if from_:
        row = storage.get_version(session, from_, version)
        if row is None:
            raise HTTPException(404, f"rule set '{from_}' not found")
        doc_json = json.dumps(row.document, indent=2)
    else:
        doc_json = json.dumps(_STARTER_DOC, indent=2)
    return _authoring_page(doc_json)


@router.post("/new", response_class=HTMLResponse)
def authoring_submit(
    doc_json: str = Form(...),
    action: str = Form(...),
    draft: str | None = Form(default=None),
    session: Session = Depends(get_session),
):
    try:
        doc = AuthoringDocument.model_validate(json.loads(doc_json))
        compiled = compile_document(doc, storage.import_resolver(session))
    except CompileError as e:
        return _authoring_page(doc_json, errors=e.errors)
    except (ValueError, json.JSONDecodeError) as e:
        return _authoring_page(doc_json, error=f"Invalid document: {e}")

    if action == "validate":
        preview = (_render_node(doc.root, "root", None)
                   + _rules_cards(compiled, None))
        return _authoring_page(doc_json, preview=preview)

    # ponytail: duplicated from main.publish_ruleset; extract if it grows
    if doc.imports:
        doc = doc.model_copy(update={"imports": [
            ImportSpec(ruleset=s.ruleset,
                       version=compiled.resolved_imports[s.ruleset])
            for s in doc.imports
        ]})
    row = storage.publish_version(
        session, doc, status="draft" if draft else "published")
    return RedirectResponse(f"/ui/{doc.name}?version={row.version}",
                            status_code=303)


@router.get("/{name}", response_class=HTMLResponse)
def document_page(
    name: str,
    version: int | None = Query(default=None),
    session: Session = Depends(get_session),
) -> str:
    row, compiled = _load(session, name, version)
    versions = storage.list_versions(session, name)
    return _document_page(row, compiled, versions)


@router.post("/{name}/evaluate", response_class=HTMLResponse)
async def evaluate_page(
    name: str,
    request: Request,
    version: int | None = Query(default=None),
    session: Session = Depends(get_session),
) -> str:
    # async + request.form(): the per-fact fields (f:<name>, s:<name>) are
    # dynamic, so they can't be declared as Form(...) parameters.
    row, compiled = _load(session, name, version)
    versions = storage.list_versions(session, name)
    form = await request.form()
    facts_json = str(form.get("facts_json") or "")
    statements_json = str(form.get("statements_json") or "")
    threshold = DEFAULT_CONFIDENCE_THRESHOLD

    # Merge order, least to most explicit: statements < raw facts JSON
    # < per-fact form rows (same direction as the JSON API).
    facts: dict[str, FactInput] = {}
    try:
        raw_thr = str(form.get("confidence_threshold") or "")
        if raw_thr:
            threshold = float(raw_thr)

        if statements_json.strip():
            stmts_raw = json.loads(statements_json)
            if not isinstance(stmts_raw, list):
                raise ValueError("statements must be a JSON array")
            stmts = [Statement.model_validate(s) for s in stmts_raw]
            facts.update(resolve_facts(stmts, threshold))

        if facts_json.strip():
            raw = json.loads(facts_json)
            if not isinstance(raw, dict):
                raise ValueError("facts must be a JSON object")
            raw = {k: v for k, v in raw.items() if v is not None}
            facts.update(coerce_facts(raw))

        for key, val in form.items():
            if not key.startswith("f:"):
                continue
            fname = key[2:]
            status = str(form.get(f"s:{fname}") or "known")
            if status != "known":
                facts[fname] = FactInput(status=FactStatus(status))
                continue
            sval = str(val).strip()
            if not sval:
                continue  # empty + known = not provided
            try:
                value = json.loads(sval)
            except json.JSONDecodeError:
                value = sval  # bare text is fine
            if value is None:
                continue
            facts[fname] = FactInput(value=value)
    except (ValueError, json.JSONDecodeError) as e:
        return _document_page(row, compiled, versions,
                              facts_json=facts_json or None,
                              statements_json=statements_json or None,
                              threshold=threshold,
                              error=f"Invalid facts: {e}")

    evaluator = Evaluator(facts, compiled.rules_by_id)
    report = evaluator.report(compiled.doc, version=row.version)
    storage.record_evaluation(
        session, row.id,
        {k: fi.model_dump(mode="json") for k, fi in facts.items()},
        report.model_dump(mode="json"),
    )
    return _document_page(row, compiled, versions, report=report,
                          facts_json=facts_json or None,
                          statements_json=statements_json or None,
                          threshold=threshold)


@router.post("/{name}/promote", response_class=HTMLResponse)
def promote_page(
    name: str,
    version: int = Query(...),
    session: Session = Depends(get_session),
):
    try:
        row = storage.promote_version(session, name, version)
    except ValueError as e:
        raise HTTPException(409, str(e))
    if row is None:
        raise HTTPException(404, f"'{name}' version {version} not found")
    return RedirectResponse(f"/ui/{name}?version={version}", status_code=303)


@router.get("/{name}/history", response_class=HTMLResponse)
def history_page(
    name: str, session: Session = Depends(get_session)
) -> str:
    if not storage.list_versions(session, name):
        raise HTTPException(404, f"rule set '{name}' not found")
    title = f"{name} — history"
    records = storage.list_evaluations(session, name)
    if not records:
        return _page(title, f"<h1>{esc(title)}</h1>"
                     "<p class='mut'>No evaluations yet.</p>")
    rows = []
    for rec, vnum in records:
        text, cls = _DECISION_LABEL[rec.report["decision"]]
        rows.append(
            f"<tr><td>{esc(rec.created_at.strftime('%Y-%m-%d %H:%M:%S'))}</td>"
            f"<td><a href='/ui/{esc(name)}?version={vnum}'>v{vnum}</a></td>"
            f"<td><span class='chip {cls}'>{esc(text)}</span></td>"
            f"<td><a href='/ui/{esc(name)}/evaluations/{rec.id}'>view</a>"
            f"</td></tr>")
    return _page(title, f"<h1>{esc(title)}</h1><table>"
                 "<tr><th>Time</th><th>Version</th><th>Decision</th><th></th>"
                 f"</tr>{''.join(rows)}</table>")


@router.get("/{name}/evaluations/{eval_id}", response_class=HTMLResponse)
def evaluation_detail_page(
    name: str, eval_id: int, session: Session = Depends(get_session)
) -> str:
    found = storage.get_evaluation(session, name, eval_id)
    if found is None:
        raise HTTPException(404, f"evaluation {eval_id} not found "
                            f"for '{name}'")
    rec, vrow = found
    doc = AuthoringDocument.model_validate(vrow.document)
    # Stored imports are pinned, so this replays against exact versions.
    compiled = compile_document(doc, storage.import_resolver(session))
    versions = storage.list_versions(session, name)
    report = EvaluationReport.model_validate(rec.report)
    when = rec.created_at.strftime('%Y-%m-%d %H:%M:%S')
    return _document_page(vrow, compiled, versions, report=report,
                          note=f"Historical evaluation #{rec.id} from {when}")


@router.get("/{name}/diff", response_class=HTMLResponse)
def diff_page(
    name: str,
    a: int = Query(...),
    b: int = Query(...),
    session: Session = Depends(get_session),
) -> str:
    ra = storage.get_version(session, name, a)
    rb = storage.get_version(session, name, b)
    if ra is None or rb is None:
        missing = a if ra is None else b
        raise HTTPException(404, f"'{name}' version {missing} not found")
    title = f"{name}: v{a} → v{b}"
    lines = list(difflib.unified_diff(
        json.dumps(ra.document, indent=2, sort_keys=True).splitlines(),
        json.dumps(rb.document, indent=2, sort_keys=True).splitlines(),
        fromfile=f"v{a}", tofile=f"v{b}", lineterm=""))
    if not lines:
        return _page(title, f"<h1>{esc(title)}</h1>"
                     "<p class='mut'>No differences.</p>")
    out = []
    for ln in lines:
        e = esc(ln)
        if ln.startswith("+++") or ln.startswith("---"):
            out.append(e)
        elif ln.startswith("+"):
            out.append(f"<span class='dadd'>{e}</span>")
        elif ln.startswith("-"):
            out.append(f"<span class='ddel'>{e}</span>")
        else:
            out.append(e)
    body = "\n".join(out)
    return _page(title, f"<h1>{esc(title)}</h1>"
                 f"<pre class='diff'>{body}</pre>")
