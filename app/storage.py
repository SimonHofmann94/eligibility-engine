"""Persistence — append-only rule versioning and an evaluation audit trail.

Versioning model: a rule set is a name; every publish creates a new immutable
`RuleSetVersion` row (version = max + 1). Nothing is ever updated in place, so
any historical decision can be replayed against the exact document that
produced it.

Runs on SQLite out of the box for the POC; set DATABASE_URL to e.g.
``postgresql+psycopg://user:pass@host/db`` for Postgres — documents are stored
as JSONB there via the type variant below. For real deployments, manage the
schema with Alembic migrations instead of ``create_all``.
"""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from .models import AuthoringDocument

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./eligibility.db")

engine = create_engine(DATABASE_URL, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class RuleSet(Base):
    __tablename__ = "rule_sets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True, index=True)

    versions: Mapped[list["RuleSetVersion"]] = relationship(
        back_populates="rule_set", order_by="RuleSetVersion.version"
    )


class RuleSetVersion(Base):
    __tablename__ = "rule_set_versions"
    __table_args__ = (UniqueConstraint("rule_set_id", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    rule_set_id: Mapped[int] = mapped_column(ForeignKey("rule_sets.id"))
    version: Mapped[int]
    # Lifecycle: 'draft' -> 'published'. Drafts can be evaluated by explicit
    # version (for testing) but are never resolved as "latest" and never
    # importable. Documents themselves stay immutable in both states.
    status: Mapped[str] = mapped_column(default="published",
                                        server_default="published")
    document: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    created_by: Mapped[str | None]

    rule_set: Mapped[RuleSet] = relationship(back_populates="versions")


class EvaluationRecord(Base):
    """Audit trail: which facts were evaluated against which version,
    and what came out."""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("rule_set_versions.id"))
    facts: Mapped[dict] = mapped_column(JSONType)
    report: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


def init_db() -> None:
    Base.metadata.create_all(engine)


# -- repository helpers -----------------------------------------------------

def publish_version(
    session: Session,
    doc: AuthoringDocument,
    created_by: str | None = None,
    status: str = "published",
) -> RuleSetVersion:
    rule_set = session.scalar(select(RuleSet).where(RuleSet.name == doc.name))
    if rule_set is None:
        rule_set = RuleSet(name=doc.name)
        session.add(rule_set)
        session.flush()

    latest = session.scalar(
        select(func.max(RuleSetVersion.version)).where(
            RuleSetVersion.rule_set_id == rule_set.id
        )
    ) or 0

    row = RuleSetVersion(
        rule_set_id=rule_set.id,
        version=latest + 1,
        status=status,
        document=doc.model_dump(mode="json"),
        created_by=created_by,
    )
    session.add(row)
    session.flush()
    return row


def get_version(
    session: Session, name: str, version: int | None = None
) -> RuleSetVersion | None:
    q = (
        select(RuleSetVersion)
        .join(RuleSet)
        .where(RuleSet.name == name)
    )
    if version is None:
        # "Latest" always means latest PUBLISHED — drafts are only reachable
        # by explicit version.
        q = (q.where(RuleSetVersion.status == "published")
              .order_by(RuleSetVersion.version.desc()).limit(1))
    else:
        q = q.where(RuleSetVersion.version == version)
    return session.scalar(q)


def promote_version(
    session: Session, name: str, version: int
) -> RuleSetVersion | None:
    """draft -> published. Returns None if not found; raises ValueError if
    the version is already published (promotion is one-way, once)."""
    row = get_version(session, name, version)
    if row is None:
        return None
    if row.status == "published":
        raise ValueError(f"'{name}' v{version} is already published")
    row.status = "published"
    session.flush()
    return row


def registry(session: Session) -> list[dict]:
    """One entry per rule set: latest published/draft versions and count."""
    entries = []
    for rs in session.scalars(select(RuleSet).order_by(RuleSet.name)):
        rows = session.scalars(
            select(RuleSetVersion)
            .where(RuleSetVersion.rule_set_id == rs.id)
        ).all()
        published = [r.version for r in rows if r.status == "published"]
        drafts = [r.version for r in rows if r.status == "draft"]
        entries.append({
            "name": rs.name,
            "latest_published": max(published) if published else None,
            "latest_draft": max(drafts) if drafts else None,
            "versions": len(rows),
            "updated_at": max(r.created_at for r in rows).isoformat()
            if rows else None,
        })
    return entries


def list_versions(session: Session, name: str) -> list[RuleSetVersion]:
    return list(
        session.scalars(
            select(RuleSetVersion)
            .join(RuleSet)
            .where(RuleSet.name == name)
            .order_by(RuleSetVersion.version)
        )
    )


def import_resolver(session: Session):
    """Resolver for the compile stage: only published versions are
    importable — a draft library must be promoted before anyone builds on
    it. Shared by the JSON API and the viewer."""
    def resolve(name: str, version: int | None):
        row = get_version(session, name, version)
        if row is None:
            raise LookupError(
                f"rule set '{name}'"
                + (f" version {version}" if version else "")
                + " not found (or has no published version)"
            )
        if row.status != "published":
            raise LookupError(
                f"'{name}' v{row.version} is not published "
                f"(status: {row.status})"
            )
        return AuthoringDocument.model_validate(row.document), row.version
    return resolve


def record_evaluation(
    session: Session, version_id: int, facts: dict, report: dict
) -> EvaluationRecord:
    rec = EvaluationRecord(version_id=version_id, facts=facts, report=report)
    session.add(rec)
    session.flush()
    return rec


def list_evaluations(session: Session, name: str, limit: int = 50):
    """Newest-first evaluation history for one rule set, as
    (EvaluationRecord, version_number) pairs."""
    # ponytail: hardcoded limit, no pagination — add offset if history grows.
    # id desc, not created_at: SQLite timestamps tie at second resolution.
    return session.execute(
        select(EvaluationRecord, RuleSetVersion.version)
        .join(RuleSetVersion,
              EvaluationRecord.version_id == RuleSetVersion.id)
        .join(RuleSet)
        .where(RuleSet.name == name)
        .order_by(EvaluationRecord.id.desc())
        .limit(limit)
    ).all()


def get_evaluation(
    session: Session, name: str, eval_id: int
) -> tuple[EvaluationRecord, RuleSetVersion] | None:
    """One evaluation plus its version row; the name join guards against
    reading another rule set's records by id."""
    row = session.execute(
        select(EvaluationRecord, RuleSetVersion)
        .join(RuleSetVersion,
              EvaluationRecord.version_id == RuleSetVersion.id)
        .join(RuleSet)
        .where(RuleSet.name == name, EvaluationRecord.id == eval_id)
    ).first()
    return (row[0], row[1]) if row else None
