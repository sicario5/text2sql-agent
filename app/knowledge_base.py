"""RAG knowledge base: schema ingestion, training and retrieval.

Four kinds of knowledge go into Vanna's vector store:

* **DDL**            one CREATE TABLE statement per approved table, with column comments
* **documentation**  table descriptions, foreign-key relationships, known
                     categorical values, and business definitions from
                     ``knowledge/business_definitions.md``
* **question/SQL**   verified example pairs from ``knowledge/sql_examples.yaml``
                     plus examples users save from the UI

At question time ``retrieve_context`` pulls the most similar items of each kind,
and the SQL generator puts them into the prompt.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .schema_inspector import SchemaInfo
from .sql_validator import SQLValidator

logger = logging.getLogger(__name__)

BUSINESS_DOC_FILE = "business_definitions.md"
SQL_EXAMPLES_FILE = "sql_examples.yaml"


@dataclass
class TrainingPlan:
    ddl: list[str] = field(default_factory=list)
    documentation: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)  # {"question": ..., "sql": ...}
    skipped_examples: list[tuple[str, str]] = field(default_factory=list)  # (question, reason)

    @property
    def size(self) -> int:
        return len(self.ddl) + len(self.documentation) + len(self.examples)


@dataclass
class RetrievedContext:
    ddl: list[str] = field(default_factory=list)
    documentation: list[str] = field(default_factory=list)
    examples: list[dict] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        return {"ddl": len(self.ddl), "documentation": len(self.documentation), "examples": len(self.examples)}


# --------------------------------------------------------------------------- building
def build_training_plan(
    schema_info: SchemaInfo,
    knowledge_dir: str | Path,
    validator: SQLValidator | None = None,
) -> TrainingPlan:
    """Collect everything that should be in the knowledge base for this schema."""
    plan = TrainingPlan()
    for table in schema_info.tables.values():
        plan.ddl.append(table.to_ddl())
        plan.documentation.append(table.to_description())

    if rel_doc := schema_info.relationships_doc():
        plan.documentation.append(rel_doc)
    plan.documentation.append(
        "Tables available for querying in schema "
        f"{schema_info.schema}: " + ", ".join(schema_info.qualified_table_names) + "."
    )

    knowledge_dir = Path(knowledge_dir)
    plan.documentation.extend(load_business_documents(knowledge_dir / BUSINESS_DOC_FILE))

    for example in load_sql_examples(knowledge_dir / SQL_EXAMPLES_FILE):
        if validator is not None:
            check = validator.validate(example["sql"])
            if not check.is_valid:
                plan.skipped_examples.append((example["question"], check.error_message))
                logger.warning("Skipping example %r: %s", example["question"], check.error_message)
                continue
        plan.examples.append(example)
    return plan


def load_business_documents(path: Path) -> list[str]:
    """Split a markdown file into one document per ``##`` section."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^##\s+", text)
    docs = []
    for section in sections[1:]:  # sections[0] is the preamble before the first '##'
        title, _, body = section.partition("\n")
        body = body.strip()
        if body:
            docs.append(f"{title.strip()}:\n{body}")
    return docs


def load_sql_examples(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    examples = []
    for item in data:
        question = str(item.get("question", "")).strip()
        sql = str(item.get("sql", "")).strip()
        if question and sql:
            examples.append({"question": question, "sql": sql})
    return examples


# --------------------------------------------------------------------------- training
def train(vn, plan: TrainingPlan, reset: bool = False) -> dict[str, int]:
    """Load the plan into Vanna's vector store. IDs are content hashes, so
    re-training is idempotent; ``reset=True`` clears stale items first."""
    if reset:
        reset_knowledge_base(vn)
    for ddl in plan.ddl:
        vn.add_ddl(ddl)
    for doc in plan.documentation:
        vn.add_documentation(doc)
    for ex in plan.examples:
        vn.add_question_sql(question=ex["question"], sql=ex["sql"])
    return {"ddl": len(plan.ddl), "documentation": len(plan.documentation), "examples": len(plan.examples)}


def reset_knowledge_base(vn) -> None:
    for collection in ("ddl", "documentation", "sql"):
        try:
            vn.remove_collection(collection)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not reset collection %s: %s", collection, exc)


def knowledge_base_counts(vn) -> dict[str, int]:
    counts = {"ddl": 0, "documentation": 0, "sql": 0}
    try:
        df = vn.get_training_data()
    except Exception:  # noqa: BLE001 - empty store
        return counts
    if df is None or df.empty or "training_data_type" not in df:
        return counts
    for kind, n in df["training_data_type"].value_counts().items():
        counts[str(kind)] = int(n)
    return counts


def is_trained(vn) -> bool:
    counts = knowledge_base_counts(vn)
    return counts["ddl"] > 0


def add_verified_example(vn, validator: SQLValidator, question: str, sql: str) -> str:
    """Store a user-approved question/SQL pair so similar questions improve."""
    check = validator.validate(sql)
    if not check.is_valid:
        raise ValueError(f"Example SQL is not valid: {check.error_message}")
    return vn.add_question_sql(question=question.strip(), sql=sql.strip())


# --------------------------------------------------------------------------- retrieval
def retrieve_context(
    vn,
    question: str,
    schema_info: SchemaInfo | None = None,
    allowed_tables: list[str] | None = None,
) -> RetrievedContext:
    """Fetch the DDL, documentation and examples most relevant to ``question``.

    When ``schema_info`` is given, the DDL of tables directly linked by a foreign
    key to a retrieved table is added too, so the LLM always sees both sides of a
    join instead of guessing the other table's columns. ``allowed_tables`` drops
    DDL for tables that are not approved, so the LLM is never shown them.
    """
    context = RetrievedContext(
        ddl=_dedupe(vn.get_related_ddl(question) or []),
        documentation=_dedupe(vn.get_related_documentation(question) or []),
        examples=[e for e in (vn.get_similar_question_sql(question) or []) if e and "sql" in e],
    )
    if schema_info is not None:
        context.ddl = expand_with_related_tables(context.ddl, schema_info)
    if allowed_tables is not None:
        allowed = {t.lower() for t in allowed_tables}
        context.ddl = [d for d in context.ddl if (_ddl_table_name(d) or "").lower() in allowed]
    return context


def _ddl_table_name(ddl: str) -> str | None:
    match = _DDL_NAME_RE.search(ddl)
    return match.group(1).replace('"', "") if match else None


_DDL_NAME_RE = re.compile(r"CREATE\s+(?:TABLE|VIEW)\s+([\w\.\"]+)", re.IGNORECASE)


def expand_with_related_tables(ddl_list: list[str], schema_info: SchemaInfo, max_extra: int = 6) -> list[str]:
    by_name = {t.full_name: t for t in schema_info.tables.values()}
    present = [name for name in (_ddl_table_name(d) for d in ddl_list) if name]
    wanted = list(present)
    for name in present:
        table = by_name.get(name)
        if not table:
            continue
        for fk in table.foreign_keys:
            wanted.append(f"{fk.referred_schema}.{fk.referred_table}")
        for other in schema_info.tables.values():  # tables that reference this one
            if any(f"{fk.referred_schema}.{fk.referred_table}" == name for fk in other.foreign_keys):
                wanted.append(other.full_name)
    extra = [by_name[n].to_ddl() for n in _dedupe(wanted) if n in by_name and n not in present][:max_extra]
    return _dedupe(ddl_list + extra)


def _dedupe(items: list[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
