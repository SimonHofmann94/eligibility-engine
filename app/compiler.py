"""The compile step — from AuthoringDocument to a validated, executable form.

This is the deck's "Compiler" stage: reference resolution, dependency
checking, and cycle detection happen here, at publish time — never as a
runtime surprise. The output (`CompiledDocument`) is the seed of the
CanonicalKnowledgeGraph/ExecutionPlan split: it carries the rule index and a
topological order of rule dependencies. The evaluator currently uses the
index with lazy memoized evaluation; the topo order is ready for an explicit
ExecutionPlan (batch pre-computation, incremental re-evaluation) later.

Compilation is pure (no storage access), deterministic, and raises
`CompileError` with *all* problems found, not just the first — authors fix a
document in one round trip.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .models import (
    AllBlock,
    AnyBlock,
    AuthoringDocument,
    ConditionalRequirement,
    NamedRule,
    NotBlock,
    OneOfBlock,
    RuleRef,
)

# Resolves (ruleset_name, pinned_version_or_None) to the document and the
# concrete version that was resolved. Must raise LookupError with a readable
# message when the target does not exist or is not importable (e.g. draft).
# Injected so the compiler itself stays pure and deterministic.
ImportResolver = Callable[[str, int | None], tuple[AuthoringDocument, int]]


class CompileError(Exception):
    """One or more compile-time problems, with author-readable messages."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class CompiledDocument:
    doc: AuthoringDocument
    rules_by_id: dict[str, NamedRule]
    dependencies: dict[str, set[str]] = field(default_factory=dict)
    topo_order: list[str] = field(default_factory=list)
    # ruleset name -> concrete version used (incl. transitive imports)
    resolved_imports: dict[str, int] = field(default_factory=dict)


def _collect_refs(node, path: str, out: list[tuple[str, str]]) -> None:
    """Walk a block tree and collect (referenced_rule_id, path) pairs."""
    if isinstance(node, RuleRef):
        out.append((node.rule, path))
    elif isinstance(node, (AllBlock, AnyBlock, OneOfBlock)):
        for i, child in enumerate(node.children):
            _collect_refs(child, f"{path}.children[{i}]", out)
    elif isinstance(node, NotBlock):
        _collect_refs(node.child, f"{path}.child", out)
    elif isinstance(node, ConditionalRequirement):
        _collect_refs(node.when, f"{path}.when", out)
        _collect_refs(node.require, f"{path}.require", out)
    # Comparison and future leaf kinds: nothing to do.


def _namespace_node(node, ns: str):
    """Rewrite an imported rule tree so its internal references become
    namespaced (``adult`` -> ``common-kyc:adult``). Already-namespaced
    references (the library's own imports) are left untouched — namespaces
    are global ruleset names, so they remain valid after merging."""
    if isinstance(node, RuleRef):
        if ":" not in node.rule:
            return node.model_copy(update={"rule": f"{ns}:{node.rule}"})
        return node
    if isinstance(node, (AllBlock, AnyBlock, OneOfBlock)):
        return node.model_copy(update={
            "children": [_namespace_node(c, ns) for c in node.children]
        })
    if isinstance(node, NotBlock):
        return node.model_copy(update={"child": _namespace_node(node.child, ns)})
    if isinstance(node, ConditionalRequirement):
        return node.model_copy(update={
            "when": _namespace_node(node.when, ns),
            "require": _namespace_node(node.require, ns),
        })
    return node


def compile_document(
    doc: AuthoringDocument,
    resolver: ImportResolver | None = None,
) -> CompiledDocument:
    errors: list[str] = []
    merged: dict[str, NamedRule] = {}
    resolved: dict[str, int] = {}
    import_stack: list[str] = [doc.name]

    def load(name: str, version: int | None, requested_by: str) -> None:
        if name in import_stack:
            errors.append(
                "circular import: " + " -> ".join(import_stack + [name])
            )
            return
        if name in resolved:
            if version is not None and resolved[name] != version:
                errors.append(
                    f"import version conflict for '{name}': "
                    f"v{resolved[name]} already in use, "
                    f"'{requested_by}' requires v{version}"
                )
            return
        if resolver is None:
            errors.append(
                f"import '{name}': document has imports but no import "
                "resolver is available"
            )
            return
        try:
            imp_doc, imp_version = resolver(name, version)
        except LookupError as e:
            errors.append(f"import '{name}': {e}")
            return
        resolved[name] = imp_version
        import_stack.append(name)
        for spec in imp_doc.imports:  # transitive imports
            load(spec.ruleset, spec.version, name)
        import_stack.pop()
        for r in imp_doc.rules:
            merged[f"{name}:{r.id}"] = r.model_copy(update={
                "id": f"{name}:{r.id}",
                "root": _namespace_node(r.root, name),
            })

    for spec in doc.imports:
        load(spec.ruleset, spec.version, doc.name)

    rules_by_id = dict(merged)
    for r in doc.rules:
        if r.id in rules_by_id:
            errors.append(f"rule id '{r.id}' collides with an imported rule")
        rules_by_id[r.id] = r

    # Resolve references (all rules — local and imported — plus the root).
    deps: dict[str, set[str]] = {rid: set() for rid in rules_by_id}
    for rid, rule in rules_by_id.items():
        refs: list[tuple[str, str]] = []
        _collect_refs(rule.root, f"rule:{rid}", refs)
        for target, path in refs:
            if target == rid:
                errors.append(f"{path}: rule '{rid}' references itself")
            elif target not in rules_by_id:
                errors.append(f"{path}: reference to unknown rule '{target}'")
            else:
                deps[rid].add(target)

    root_refs: list[tuple[str, str]] = []
    _collect_refs(doc.root, "root", root_refs)
    for target, path in root_refs:
        if target not in rules_by_id:
            errors.append(f"{path}: reference to unknown rule '{target}'")

    # 2. Cycle detection + topological order (DFS, deterministic).
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {rid: WHITE for rid in rules_by_id}
    topo: list[str] = []
    stack: list[str] = []

    def visit(rid: str) -> None:
        color[rid] = GRAY
        stack.append(rid)
        for dep in sorted(deps.get(rid, ())):
            if color[dep] == GRAY:
                cycle = stack[stack.index(dep):] + [dep]
                errors.append(
                    "circular rule dependency: " + " -> ".join(cycle)
                )
            elif color[dep] == WHITE:
                visit(dep)
        stack.pop()
        color[rid] = BLACK
        topo.append(rid)  # dependencies first

    for rid in sorted(rules_by_id):
        if color[rid] == WHITE:
            visit(rid)

    if errors:
        raise CompileError(errors)

    return CompiledDocument(
        doc=doc, rules_by_id=rules_by_id, dependencies=deps,
        topo_order=topo, resolved_imports=resolved,
    )
