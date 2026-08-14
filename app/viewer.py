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

import html
import json

from fastapi import APIRouter, Depends, Form, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from . import storage
from .compiler import CompiledDocument, compile_document
from .evaluator import _OPS, Evaluator
from .models import (
    AllBlock,
    AnyBlock,
    AuthoringDocument,
    Comparison,
    ConditionalRequirement,
    NotBlock,
    OneOfBlock,
    RuleRef,
    coerce_facts,
)
from .report import STATUS_TO_DECISION, EvaluationReport, RuleResult

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


def _document_page(
    row,
    compiled: CompiledDocument,
    all_versions,
    report: EvaluationReport | None = None,
    facts_json: str | None = None,
    error: str | None = None,
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
    if doc.description:
        parts.append(f"<p class='mut'>{esc(doc.description)}</p>")

    if error:
        parts.append(f"<div class='err'>{esc(error)}</div>")

    if report:
        text, cls = _DECISION_LABEL[report.decision.value]
        parts.append(
            f"<div class='banner'><span>Decision</span>"
            f"<span class='chip {cls}'>{esc(text)}</span></div>"
        )

    # Facts form — prefilled with a skeleton of every fact the rules read;
    # null means "not provided" and evaluates as unknown.
    if facts_json is None:
        skeleton = {name: None for name in collect_fact_names(compiled)}
        facts_json = json.dumps(skeleton, indent=1)
    parts.append(
        f"<div class='card'><div class='hd'>Facts</div>"
        f"<div class='mut'>null = not provided → evaluates as unknown</div>"
        f"<form method='post' "
        f"action='/ui/{esc(doc.name)}/evaluate?version={row.version}'>"
        f"<textarea name='facts_json'>{esc(facts_json)}</textarea>"
        f"<button type='submit'>Evaluate v{row.version}</button></form></div>"
    )

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

    if compiled.rules_by_id:
        parts.append("<h2>Named rules</h2>")
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
def evaluate_page(
    name: str,
    facts_json: str = Form(...),
    version: int | None = Query(default=None),
    session: Session = Depends(get_session),
) -> str:
    row, compiled = _load(session, name, version)
    versions = storage.list_versions(session, name)

    try:
        raw = json.loads(facts_json)
        if not isinstance(raw, dict):
            raise ValueError("facts must be a JSON object")
        raw = {k: v for k, v in raw.items() if v is not None}
        facts = coerce_facts(raw)
    except (ValueError, json.JSONDecodeError) as e:
        return _document_page(row, compiled, versions,
                              facts_json=facts_json, error=f"Invalid facts: {e}")

    evaluator = Evaluator(facts, compiled.rules_by_id)
    report = evaluator.report(compiled.doc, version=row.version)
    storage.record_evaluation(
        session, row.id, raw, report.model_dump(mode="json")
    )
    return _document_page(row, compiled, versions, report=report,
                          facts_json=facts_json)
