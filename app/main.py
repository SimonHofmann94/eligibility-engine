"""API layer.

Validation at the edge: publishing a document runs it through the typed
Pydantic models, so invalid documents are rejected with clear, field-level
messages (FastAPI 422) before they ever reach storage — a lightweight version
of the deck's "Compiler" stage. Reference resolution / dependency checking
joins this stage once calculations and rule references are added.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Iterator

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .compiler import CompileError, compile_document
from .evaluator import evaluate_document
from .models import AuthoringDocument, ImportSpec, coerce_facts
from .report import EvaluationReport
from .statements import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    Statement,
    resolve_facts,
)
from . import storage
from .viewer import router as viewer_router


def _compile_or_422(doc: AuthoringDocument, session: Session):
    """Structure is checked by Pydantic; this adds the compile stage —
    reference and import resolution, cycle detection — reporting all errors
    at once."""
    try:
        return compile_document(doc, storage.import_resolver(session))
    except CompileError as e:
        raise HTTPException(status_code=422, detail={"compile_errors": e.errors})


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()  # POC convenience; use Alembic in production
    yield


app = FastAPI(
    title="Eligibility Rule Engine",
    version="0.5",
    description="Applicability gates, four-state evaluation, "
    "versioned rule sets.",
    lifespan=lifespan,
)
app.include_router(viewer_router)


def get_session() -> Iterator[Session]:
    with storage.SessionLocal() as session:
        with session.begin():
            yield session


# -- response/request shapes ------------------------------------------------

class PublishResponse(BaseModel):
    name: str
    version: int
    status: str
    resolved_imports: dict[str, int]


class VersionInfo(BaseModel):
    version: int
    status: str
    created_at: str
    created_by: str | None


class RegistryEntry(BaseModel):
    name: str
    latest_published: int | None
    latest_draft: int | None
    versions: int
    updated_at: str | None


class EvaluateRequest(BaseModel):
    """Either raw facts, extracted statements, or both (explicit facts win
    per field over resolved statements)."""

    facts: dict[str, Any] | None = None
    statements: list[Statement] | None = None
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    extractor_version: str | None = None


class ValidateResponse(BaseModel):
    valid: bool
    normalized: AuthoringDocument


# -- endpoints ---------------------------------------------------------------

@app.post("/rulesets", response_model=PublishResponse, status_code=201)
def publish_ruleset(
    doc: AuthoringDocument,
    draft: bool = Query(default=False, description="Create as draft — "
                        "testable by explicit version, not resolved as "
                        "latest, not importable until promoted"),
    session: Session = Depends(get_session),
) -> PublishResponse:
    """Create a new immutable version of a rule set (append-only).

    The document must compile: unknown references, circular dependencies,
    and unimportable libraries are rejected here, never at evaluation time.
    Pin-on-publish: imports submitted without a version are resolved to the
    latest published library version and that pin is frozen into the stored
    document, so every future evaluation replays against exact versions."""
    compiled = _compile_or_422(doc, session)
    if doc.imports:
        doc = doc.model_copy(update={"imports": [
            ImportSpec(ruleset=s.ruleset,
                       version=compiled.resolved_imports[s.ruleset])
            for s in doc.imports
        ]})
    row = storage.publish_version(
        session, doc, status="draft" if draft else "published"
    )
    return PublishResponse(name=doc.name, version=row.version,
                           status=row.status,
                           resolved_imports=compiled.resolved_imports)


@app.get("/rulesets", response_model=list[RegistryEntry])
def list_rulesets(session: Session = Depends(get_session)) -> list[RegistryEntry]:
    """The registry: one entry per credit process / rule set."""
    return [RegistryEntry(**e) for e in storage.registry(session)]


@app.post("/rulesets/{name}/versions/{version}/publish",
          response_model=VersionInfo)
def promote(
    name: str, version: int, session: Session = Depends(get_session)
) -> VersionInfo:
    """Promote a draft to published (one-way)."""
    try:
        row = storage.promote_version(session, name, version)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if row is None:
        raise HTTPException(404, f"'{name}' version {version} not found")
    return VersionInfo(version=row.version, status=row.status,
                       created_at=row.created_at.isoformat(),
                       created_by=row.created_by)


@app.post("/rulesets/validate", response_model=ValidateResponse)
def validate_ruleset(
    doc: AuthoringDocument, session: Session = Depends(get_session)
) -> ValidateResponse:
    """Validate, compile, and normalize a document without persisting it."""
    _compile_or_422(doc, session)
    return ValidateResponse(valid=True, normalized=doc)


@app.get("/rulesets/{name}/versions", response_model=list[VersionInfo])
def get_versions(
    name: str, session: Session = Depends(get_session)
) -> list[VersionInfo]:
    rows = storage.list_versions(session, name)
    if not rows:
        raise HTTPException(404, f"rule set '{name}' not found")
    return [
        VersionInfo(
            version=r.version,
            status=r.status,
            created_at=r.created_at.isoformat(),
            created_by=r.created_by,
        )
        for r in rows
    ]


@app.get("/rulesets/{name}/versions/{version}", response_model=AuthoringDocument)
def get_document(
    name: str, version: int, session: Session = Depends(get_session)
) -> AuthoringDocument:
    row = storage.get_version(session, name, version)
    if row is None:
        raise HTTPException(404, f"'{name}' version {version} not found")
    return AuthoringDocument.model_validate(row.document)


@app.post("/rulesets/{name}/evaluate", response_model=EvaluationReport)
def evaluate(
    name: str,
    body: EvaluateRequest,
    version: int | None = Query(default=None, description="Pin a version; "
                                "defaults to latest"),
    session: Session = Depends(get_session),
) -> EvaluationReport:
    """Evaluate facts against a rule set version and persist the report."""
    row = storage.get_version(session, name, version)
    if row is None:
        raise HTTPException(404, f"rule set '{name}' "
                            f"{'version ' + str(version) if version else ''} "
                            "not found")
    if body.facts is None and body.statements is None:
        raise HTTPException(422, "provide 'facts', 'statements', or both")
    doc = AuthoringDocument.model_validate(row.document)

    facts = {}
    if body.statements:
        facts.update(resolve_facts(body.statements,
                                   body.confidence_threshold))
    if body.facts:
        facts.update(coerce_facts(body.facts))  # explicit facts win

    extractor_version = body.extractor_version
    if extractor_version is None and body.statements:
        versions = {s.extractor_version for s in body.statements
                    if s.extractor_version}
        if len(versions) == 1:
            extractor_version = versions.pop()

    report = evaluate_document(doc, facts, version=row.version,
                               resolver=storage.import_resolver(session),
                               extractor_version=extractor_version)
    storage.record_evaluation(
        session, row.id,
        {"facts": body.facts, "statements": [
            s.model_dump(mode="json") for s in (body.statements or [])]},
        report.model_dump(mode="json"),
    )
    return report
