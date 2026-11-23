"""Natural-language -> SQL generation.

Responsibilities:

* turn a follow-up ("Now only for 2025") into a standalone question using the
  conversation history, so retrieval finds the right context;
* build the prompt: rules, today's date, approved tables, retrieved DDL /
  documentation / examples, and the recent conversation;
* call the LLM through Vanna's chat interface and extract the SQL;
* build a repair prompt when a query fails validation or execution.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field

from .knowledge_base import RetrievedContext

logger = logging.getLogger(__name__)

CANNOT_ANSWER = "CANNOT_ANSWER"

SYSTEM_TEMPLATE = """You are an expert PostgreSQL data analyst. You translate business questions into ONE correct, read-only PostgreSQL query.

Today's date is {today} ({weekday}). Use CURRENT_DATE and date_trunc() for relative periods such as "last month".

Rules:
1. Reply with exactly one SELECT statement (WITH ... SELECT is fine) inside a ```sql code block and nothing else.
2. Use ONLY tables and columns that appear in the schema context below. Never invent table or column names.
3. Approved tables: {tables}. Never reference any other table, schema or system catalog.
4. Never write INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, GRANT or anything else that changes data or schema. If the user asks to change data, reply with: {cannot}: this assistant is read-only.
5. Follow the business definitions in the documentation exactly (for example, how revenue is calculated).
6. Use table aliases and qualify every column when joining. Join tables only through the foreign-key relationships shown.
7. Give every computed column a readable snake_case alias. Round money to 2 decimals.
8. Lists of rows must have a meaningful ORDER BY and a LIMIT of at most {max_rows}. Pure aggregates do not need LIMIT.
9. For time series, return the period as the first column and order by it.
10. If the question is a follow-up, interpret it using the conversation so far and adapt the previous query.
11. If the question cannot be answered with this database, reply with exactly: {cannot}: <short reason>
"""

REWRITE_SYSTEM = """You rewrite follow-up questions for a SQL analytics assistant.
Given the conversation so far and the latest user question, rewrite the latest question so it is fully self-contained:
keep every metric, filter, grouping and time period that the follow-up implicitly carries over from earlier turns.
If the latest question starts a new topic, return it unchanged.
Return ONLY the rewritten question, with no explanation and no quotes."""


@dataclass
class Turn:
    """One completed question/answer exchange, used as conversational context."""

    question: str
    standalone_question: str
    sql: str | None = None
    columns: list[str] = field(default_factory=list)
    row_count: int | None = None


@dataclass
class GenerationResult:
    sql: str | None
    raw_response: str
    prompt: list[dict]
    cannot_answer_reason: str | None = None

    @property
    def cannot_answer(self) -> bool:
        return self.cannot_answer_reason is not None


class SQLGenerator:
    def __init__(
        self,
        vn,
        allowed_tables: list[str],
        max_rows: int = 1000,
        history_turns: int = 5,
        context_token_budget: int = 12000,
        today: dt.date | None = None,
    ) -> None:
        self.vn = vn
        self.allowed_tables = allowed_tables
        self.max_rows = max_rows
        self.history_turns = history_turns
        self.context_token_budget = context_token_budget
        self._today = today

    @property
    def today(self) -> dt.date:
        return self._today or dt.date.today()

    # ----------------------------------------------------------- follow-ups
    def rewrite_question(self, question: str, history: list[Turn]) -> str:
        """Return a standalone version of ``question`` given the conversation."""
        recent = self._recent(history)
        if not recent:
            return question
        lines = []
        for i, turn in enumerate(recent, 1):
            lines.append(f"{i}. User: {turn.question}")
            if turn.standalone_question and turn.standalone_question != turn.question:
                lines.append(f"   (interpreted as: {turn.standalone_question})")
            if turn.sql:
                lines.append(f"   SQL: {_one_line(turn.sql)}")
        prompt = [
            self.vn.system_message(REWRITE_SYSTEM),
            self.vn.user_message("Conversation:\n" + "\n".join(lines) + f"\n\nLatest question: {question}"),
        ]
        try:
            rewritten = (self.vn.submit_prompt(prompt) or "").strip().strip('"').strip()
        except Exception as exc:  # noqa: BLE001 - fall back to the raw question
            logger.warning("Question rewrite failed, using original: %s", exc)
            return question
        if not rewritten or len(rewritten) > 1000 or rewritten.upper().startswith(("SELECT", "WITH ")):
            return question
        return rewritten.splitlines()[0].strip()

    # ----------------------------------------------------------- generation
    def build_prompt(self, question: str, context: RetrievedContext, history: list[Turn]) -> list[dict]:
        system = SYSTEM_TEMPLATE.format(
            today=self.today.isoformat(),
            weekday=self.today.strftime("%A"),
            tables=", ".join(self.allowed_tables) or "(none)",
            max_rows=self.max_rows,
            cannot=CANNOT_ANSWER,
        )
        # Vanna's helpers append each block while respecting the token budget.
        system = self.vn.add_ddl_to_prompt(system, list(context.ddl), max_tokens=self.context_token_budget)
        system = self.vn.add_documentation_to_prompt(
            system, list(context.documentation), max_tokens=self.context_token_budget
        )
        system = self._add_examples(system, context.examples)

        messages = [self.vn.system_message(system)]
        for turn in self._recent(history):
            messages.append(self.vn.user_message(turn.question))
            answer = f"```sql\n{turn.sql.strip()}\n```" if turn.sql else "(no query)"
            if turn.row_count is not None:
                answer += f"\nResult: {turn.row_count} rows; columns: {', '.join(turn.columns)}"
            messages.append(self.vn.assistant_message(answer))
        user_text = question
        messages.append(self.vn.user_message(user_text))
        return messages

    def generate(self, question: str, context: RetrievedContext, history: list[Turn] | None = None) -> GenerationResult:
        prompt = self.build_prompt(question, context, history or [])
        return self._submit(prompt)

    def repair(
        self,
        question: str,
        context: RetrievedContext,
        history: list[Turn] | None,
        failed_sql: str,
        error: str,
    ) -> GenerationResult:
        """Ask the LLM to fix ``failed_sql`` given the validation/database ``error``."""
        prompt = self.build_prompt(question, context, history or [])
        prompt.append(self.vn.assistant_message(f"```sql\n{failed_sql.strip()}\n```"))
        prompt.append(
            self.vn.user_message(
                "That query failed with this error:\n"
                f"{error.strip()[:1500]}\n\n"
                "Analyse the error and return a corrected query that answers the original question: "
                f'"{question}". Use only the approved tables and the columns shown in the schema context. '
                f"Reply with the corrected query in a ```sql block, or {CANNOT_ANSWER}: <reason> if it cannot be fixed."
            )
        )
        return self._submit(prompt)

    # ----------------------------------------------------------- internals
    def _submit(self, prompt: list[dict]) -> GenerationResult:
        raw = self.vn.submit_prompt(prompt) or ""
        reason = parse_cannot_answer(raw)
        if reason is not None:
            return GenerationResult(sql=None, raw_response=raw, prompt=prompt, cannot_answer_reason=reason)
        sql = extract_sql(raw)
        if not sql:
            return GenerationResult(
                sql=None,
                raw_response=raw,
                prompt=prompt,
                cannot_answer_reason="The model did not return a SQL query.",
            )
        return GenerationResult(sql=sql, raw_response=raw, prompt=prompt)

    def _recent(self, history: list[Turn]) -> list[Turn]:
        if self.history_turns <= 0:
            return []
        return [t for t in history if t.sql][-self.history_turns :]

    def _add_examples(self, system: str, examples: list[dict]) -> str:
        if not examples:
            return system
        budget_chars = self.context_token_budget * 4  # ~4 characters per token
        block = "\n===Example questions with correct SQL (follow their style and business logic)\n"
        used = len(system)
        for ex in examples:
            entry = f"\nQuestion: {ex['question']}\n```sql\n{ex['sql'].strip()}\n```\n"
            if used + len(block) + len(entry) > budget_chars:
                break
            block += entry
        return system + block


_CODE_BLOCK_RE = re.compile(r"```[ \t]*(?:sql|postgresql|postgres|pgsql)?[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_CANNOT_RE = re.compile(rf"^\s*{CANNOT_ANSWER}\s*:?\s*(.*)", re.IGNORECASE | re.DOTALL)
_SQL_START_RE = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)
_OTHER_SQL_START_RE = re.compile(
    r"^(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|GRANT|REVOKE|MERGE|COPY|CALL|DO|SET|VACUUM)\b", re.IGNORECASE
)


def parse_cannot_answer(text: str) -> str | None:
    match = _CANNOT_RE.match(text or "")
    if not match:
        return None
    return match.group(1).strip().splitlines()[0] if match.group(1).strip() else "The question cannot be answered."


def extract_sql(text: str) -> str | None:
    """Pull the SQL query out of an LLM response."""
    if not text:
        return None
    blocks = [b.strip() for b in _CODE_BLOCK_RE.findall(text) if b.strip()]
    candidate = None
    for block in reversed(blocks):  # the last block is usually the final answer
        if _SQL_START_RE.search(block):
            candidate = block
            break
    if candidate is None and blocks:
        # A non-SELECT statement (e.g. DELETE) is still returned so the validator
        # can block it visibly instead of it being silently dropped.
        candidate = blocks[-1]
    if candidate is None:
        stripped = text.strip()
        if _OTHER_SQL_START_RE.match(stripped):
            candidate = stripped
            match = None
        else:
            match = _SQL_START_RE.search(stripped)
            if not match:
                return None
        if match is not None:
            candidate = stripped[match.start() :]
            # cut trailing prose after the statement terminator
            if ";" in candidate:
                candidate = candidate[: candidate.rindex(";") + 1]
    candidate = candidate.strip()
    while candidate.endswith(";"):
        candidate = candidate[:-1].rstrip()
    return candidate or None


def _one_line(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()
