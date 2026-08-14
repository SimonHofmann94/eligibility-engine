# Eligibility Rule Engine — v0.1 reference implementation

A lean, correct core of the design in *Eligibility Rule Modeling — Design
Recommendation*: facts with explicit statuses, applicability gates, four-state
evaluation, versioned rule sets, and reviewer-readable explanation reports.
FastAPI + Pydantic v2 + SQLAlchemy 2.

```
app/
  models.py      AuthoringDocument — the small block vocabulary (Pydantic, discriminated union on `kind`)
  evaluator.py   The four-state engine: gates, all/any/not, short-circuiting
  report.py      RuleResult / EvaluationReport, applicability-vs-truth split, decision mapping
  storage.py     Append-only rule versioning + evaluation audit trail (SQLite POC / Postgres-ready)
  main.py        API layer, validation at the edge
tests/
  test_worked_examples.py    Reproduces the deck's worked-examples table row by row
  test_semantics_and_api.py  Group/not semantics, fact statuses, API round-trip
```

## Run it

```bash
pip install -r requirements.txt
pytest                      # 21 tests, incl. the deck's worked examples
uvicorn app.main:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

```bash
# Publish (creates version 1; publishing again creates version 2, append-only)
curl -X POST localhost:8000/rulesets -H 'content-type: application/json' -d '{
  "name": "retail-eligibility",
  "root": {
    "kind": "all",
    "children": [
      {"kind": "comparison", "fact": "age", "operator": "ge", "value": 28},
      {"kind": "comparison", "fact": "residence", "operator": "eq", "value": "Germany"},
      {"kind": "conditional_requirement",
       "label": "Debt-free for higher salaries",
       "when":    {"kind": "comparison", "fact": "salary", "operator": "ge", "value": 50000},
       "require": {"kind": "comparison", "fact": "debt_free", "operator": "eq", "value": true}}
    ]
  }
}'

# Evaluate against the latest version (facts accept shorthand or explicit status)
curl -X POST localhost:8000/rulesets/retail-eligibility/evaluate \
  -H 'content-type: application/json' \
  -d '{"facts": {"age": 34, "residence": "Germany", "salary": {"status": "unknown"}, "debt_free": true}}'
# -> decision: needs_review, gate status: unknown ("it is unknown whether the requirement applies")

# Replay an old policy
curl -X POST 'localhost:8000/rulesets/retail-eligibility/evaluate?version=1' ...
```

## Design decisions

**The status enum is the contract.** Everything hangs off two enums:
`FactStatus` (known / unknown / not_applicable / error) and `RuleStatus`
(satisfied / failed / unknown / not_applicable / error / not_evaluated).
`not_evaluated` is deliberately an execution state, not a truth state — it
appears only when short-circuiting skips a branch, so the report stays honest
about what was and was not checked.

**Applicability ≠ truth.** `conditional_requirement` results carry an
`(applicability, truth)` pair alongside the combined UI status, exactly
mirroring the deck's mapping table. "The rule does not apply" (`when` failed)
and "it is unknown whether the rule applies" (`when` unknown) produce
different results.

**Missing is never false.** An absent or unknown fact makes the reading rule
`unknown`, which propagates upward and maps to a `needs_review` decision —
never a silent rejection.

**Evaluation is pure.** `Evaluator` takes facts, returns a report, touches
nothing. Storage and API are thin layers around it, so the engine is trivially
unit-testable and reusable (batch jobs, embedded use, a future ExecutionPlan).

**Validation at the edge = compiler-light.** The Pydantic discriminated union
rejects malformed documents with field-level 422s before persistence. The
deck's full Compiler stage (reference resolution, dependency wiring, cycle
detection) slots in here once calculations and rule references exist — the
API shape doesn't change.

**Append-only versioning.** A publish always creates `version = max + 1`;
rows are immutable, evaluations record the version they used, and old
decisions can be replayed bit-for-bit. This is the audit substrate that
overrides and regulatory review later build on.

### Semantics choices the deck leaves open (documented, not guessed)

1. **Error precedence in groups.** The deck's all/any cascades don't mention
   `error`. We rank it directly after the decisive state (`failed` for `all`,
   `satisfied` for `any`), so infrastructure failures are never masked by
   unknowns, but a conclusive business outcome still wins.
2. **A `not_applicable` input fact.** The fact-status slide says the requiring
   rule becomes `not_applicable`; the "important fact-handling rule" slide
   says it evaluates as `unknown` ("not_applicable is never inferred just
   because data is missing"). These conflict. Default is the conservative
   `unknown` reading; `Evaluator(na_input_policy="not_applicable")` selects
   the other. Worth resolving with the client before v0.2.

## Postgres

`DATABASE_URL=postgresql+psycopg://user:pass@host/db` switches the backend;
documents and reports are stored as JSONB via a type variant (SQLite keeps
plain JSON for the POC). Before production: replace `create_all` with Alembic
migrations, add a GIN index on `rule_set_versions.document` if you want to
query inside documents, and consider partitioning `evaluations` by month —
it's an append-only audit table and will dominate row counts.

## Multi-rule documents (v0.2)

Built for rule systems with 50–100 rules per process: a document now carries a
flat catalog of named rules plus a decision root that references them.

```jsonc
{
  "name": "consumer-credit-de",
  "rules": [
    {"id": "adult",        "root": {"kind": "comparison", "fact": "age", "operator": "ge", "value": 18}},
    {"id": "resident",     "root": {"kind": "comparison", "fact": "residence", "operator": "eq", "value": "Germany"}},
    {"id": "kyc_complete", "root": {"kind": "all", "children": [
        {"kind": "rule_ref", "rule": "adult"},
        {"kind": "rule_ref", "rule": "resident"}]}}
  ],
  "root": {"kind": "rule_ref", "rule": "kyc_complete"}
}
```

* `rule_ref` lets a rule depend on the outcome of another rule; dependencies
  are discovered from references — authors never wire them by hand.
* `app/compiler.py` is the compile stage: reference resolution, cycle
  detection (a circular dependency is a publish-time 422, never a runtime
  surprise), and a topological dependency order — reported all at once.
* Evaluation is lazy and memoized: each named rule runs at most once per
  request regardless of how many places reference it, and rules never reached
  (short-circuit) are honestly absent from the report.
* `EvaluationReport.rule_results` holds each evaluated rule's full tree
  exactly once; `rule_ref` nodes in the decision tree carry the status plus a
  `ref` pointer instead of duplicating subtrees — the report is a DAG.

## Shared libraries, lifecycle, registry (v0.3)

Built for 144 processes that share common rules without 144 diverging copies.

**Imports.** A document can import another rule set's named rules; they become
available under the namespace `<ruleset>:<rule_id>`:

```jsonc
{
  "name": "consumer-credit-de",
  "imports": [{"ruleset": "common-kyc"}],          // version optional
  "root": {"kind": "rule_ref", "rule": "common-kyc:complete"}
}
```

Imports resolve recursively (libraries may import libraries); internal
references inside imported rules are rewritten into the namespace
automatically. Cross-document circular imports and version conflicts (two
paths through the import graph demanding different versions of one library)
are compile-time 422s. The compiler stays pure — storage access is injected
as a resolver — so import semantics are testable without a database.

**Pin-on-publish.** An import without a version resolves to the latest
*published* library version, and that pin is frozen into the stored document.
Publishing `common-kyc` v2 never silently changes any process that pinned v1;
a process adopts the new library version only when it is itself re-published.
Every historical decision stays replayable bit-for-bit.

**Draft → published lifecycle.** `POST /rulesets?draft=true` creates a draft:
testable via explicit `?version=N` evaluation, but never resolved as
"latest" and never importable. `POST /rulesets/{name}/versions/{v}/publish`
promotes it (one-way, once — 409 on repeat). Both states are immutable; a
change is always a new version.

**Registry.** `GET /rulesets` lists every process with latest published
version, latest draft, and version count — the overview endpoint for a
144-process estate.

## Read-only viewer (v0.4)

`/ui` — server-rendered HTML straight from FastAPI, zero frontend build.
Registry page (all processes), document pages (nested blocks, named-rule
catalog incl. imported namespaced rules, pinned imports), and an evaluation
form: paste facts as JSON (`null` = not provided → unknown), get the decision
banner plus a status chip and reason on every node. UI evaluations land in
the same audit trail as API evaluations.

The overlay works by the path contract: the viewer walks the document with
the evaluator's exact path convention and joins results by path lookup —
lazily skipped rules render honestly as "not evaluated". The viewer is
read-only by design; the authoring UI remains a separate project fed by a
catalog endpoint.

## Statement layer + quick wins (v0.5)

Aligns the engine with the Dokumentenprüfung reference concept (extraction vs.
deterministic evaluation).

**Statements (Aussagen).** Extraction systems submit value claims with
provenance and confidence — `{field, value, confidence, source_document,
source_location, extractor_version}` — instead of finished facts.
Contradictory statements per field are allowed and wanted. Two resolvers per
the concept: `resolve_value` (override > highest confidence > most recent;
below-threshold and ambiguity → unknown, never a guess, with the reason in
the fact's note that flows into rule reasons) and `resolve_set` (distinct
values for future consistency rules). Overrides are statements with mandatory
attribution. `/evaluate` accepts `statements`, `facts`, or both — explicit
facts win per field. Subject scoping arrives with the entity step.

**`one_of` group mode.** Exactly-one semantics with a documented deviation
from the concept paper: one satisfied child plus an unknown child yields
*unknown*, not satisfied — the unknown child could resolve to satisfied and
flip the result into a conflict (no false greens). ≥2 satisfied is a
conflict and decisive regardless of unknowns.

**Stamps.** Reports and audit records now carry `engine_version` and
`extractor_version` (auto-derived when all statements agree) alongside the
ruleset version — the concept's three-version reproducibility tuple.
`NamedRule.automation` (`automatic | assisted | manual`) flows into rule
results so report consumers can route assisted/manual outcomes to review.

## Extension roadmap (in dependency order)

**1. Calculations / derived facts** — add a `calculations` list to
`AuthoringDocument` (`{"name": "annual_salary", "inputs": [...], "expr": ...}`).
A compile step (the real Compiler stage) topologically sorts calculations by
reads/writes, rejects cycles at publish time, and the evaluator materializes
derived facts into the same `FactPayload` before rules run — a derived fact is
a `FactInput` like any other, so unknown inputs automatically yield unknown
derived facts and the propagation rules need no changes. This step introduces
the CanonicalKnowledgeGraph/ExecutionPlan split for real.

**2. Decision tables** — a `decision_table` block is pure authoring sugar:
the compile step expands each row into a `conditional_requirement` /
prioritized outcome candidate. Because expansion happens at publish time, the
evaluator never learns a new node kind. Add `priority` + `default` to model
the deck's deterministic resolution (highest-priority satisfied row wins,
losers recorded in the report).

**3. Entity relationships + quantifiers** — replace flat fact names with
paths (`customer.age`, `co_applicants[*].debt_free`) and add one block:
`{"kind": "quantifier", "mode": "all"|"any", "over": "co_applicants",
"predicate": {...}}`. The quantifier binds each related entity and reuses the
existing all/any group semantics verbatim — "all co-applicants are debt-free"
is just an `all` over a collection. Facts become entity-scoped; the engine
stays domain-agnostic.

**4. Manual overrides** — a separate `overrides` table
(`evaluation_id | target_path | pinned_status_or_value | who | when | why`)
plus an override layer in the evaluator: where an override exists it wins for
downstream evaluation, but the report keeps `computed` and `overridden`
side by side, so a decision is never silently changed. Requires nothing from
the core except that `RuleResult.path` is stable — which is why paths exist
from day one.

Ordering rationale: calculations force the compiler into existence; decision
tables ride on that compiler; quantifiers need the entity-scoped fact model;
overrides only need stable paths and the audit trail, so they can land any
time after v0.1.
