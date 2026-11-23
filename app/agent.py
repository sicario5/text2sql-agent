"""The Text-to-SQL agent: orchestrates one question end to end.

question
  -> rewrite follow-up into a standalone question (uses history)
  -> retrieve schema / docs / examples (RAG)
  -> generate SQL (LLM)
  -> validate (safety layer)  --invalid but fixable--> repair (LLM) --+
  -> execute (read-only)      --database error-------> repair (LLM) --+  (max_retries)
  -> summarise + recommend chart
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from .executor import QueryExecutionError, QueryExecutor
from .knowledge_base import RetrievedContext, retrieve_context
from .llm import LLMError
from .results import fallback_summary, summarize_result
from .schema_inspector import SchemaInfo
from .sql_generator import GenerationResult, SQLGenerator, Turn
from .sql_validator import SQLValidator
from .visualization import ChartSpec, recommend_chart

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


@dataclass
class Attempt:
    number: int
    sql: str | None
    stage: str  # "generation" | "validation" | "execution" | "success"
    error: str | None = None


@dataclass
class AgentResponse:
    question: str
    standalone_question: str
    status: str  # "success" | "error" | "blocked" | "cannot_answer"
    sql: str | None = None
    df: pd.DataFrame | None = None
    row_count: int = 0
    truncated: bool = False
    execution_ms: float = 0.0
    total_ms: float = 0.0
    summary: str = ""
    chart: ChartSpec | None = None
    error: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    context_counts: dict[str, int] = field(default_factory=dict)
    limit_applied: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "success"

    def to_turn(self) -> Turn:
        return Turn(
            question=self.question,
            standalone_question=self.standalone_question,
            sql=self.sql if self.ok else None,
            columns=list(self.df.columns) if self.df is not None else [],
            row_count=self.row_count if self.ok else None,
        )


class TextToSQLAgent:
    def __init__(
        self,
        vn,
        generator: SQLGenerator,
        validator: SQLValidator,
        executor: QueryExecutor,
        schema_info: SchemaInfo | None = None,
        max_retries: int = 2,
        summary_enabled: bool = True,
        summary_max_rows: int = 50,
    ) -> None:
        self.vn = vn
        self.generator = generator
        self.validator = validator
        self.executor = executor
        self.schema_info = schema_info
        self.max_retries = max_retries
        self.summary_enabled = summary_enabled
        self.summary_max_rows = summary_max_rows

    def ask(
        self,
        question: str,
        history: list[Turn] | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> AgentResponse:
        started = time.perf_counter()
        progress = on_progress or (lambda _msg: None)
        history = history or []
        question = (question or "").strip()
        response = AgentResponse(question=question, standalone_question=question, status="error")
        if not question:
            response.error = "Please enter a question."
            return response

        try:
            # 1. follow-up resolution
            if history:
                progress("Understanding the follow-up question…")
                response.standalone_question = self.generator.rewrite_question(question, history)

            # 2. retrieval
            progress("Retrieving relevant schema and business context…")
            context = retrieve_context(
                self.vn, response.standalone_question, self.schema_info, self.validator.allowed_table_list
            )
            response.context_counts = context.counts

            # 3-5. generate -> validate -> execute, with repair attempts
            self._generate_and_run(response, context, history, progress)
        except LLMError as exc:
            response.status = "error"
            response.error = f"The language model request failed: {exc}"
        except Exception as exc:
            logger.exception("Unexpected agent failure")
            response.status = "error"
            response.error = f"Unexpected error: {exc}"

        # 6. summary + chart
        if response.ok and response.df is not None:
            progress("Summarising results…")
            response.chart = recommend_chart(response.df, response.standalone_question)
            if self.summary_enabled:
                result_view = _ResultView(response)
                response.summary = summarize_result(
                    self.vn, response.standalone_question, response.sql or "", result_view, self.summary_max_rows
                )
            else:
                response.summary = fallback_summary(_ResultView(response))

        response.total_ms = (time.perf_counter() - started) * 1000
        return response

    # ------------------------------------------------------------------ core loop
    def _generate_and_run(
        self,
        response: AgentResponse,
        context: RetrievedContext,
        history: list[Turn],
        progress: ProgressCallback,
    ) -> None:
        question = response.standalone_question
        failed_sql: str | None = None
        last_error: str | None = None

        for attempt_no in range(1, self.max_retries + 2):
            if failed_sql is None:
                progress("Generating SQL…")
                gen: GenerationResult = self.generator.generate(question, context, history)
            else:
                progress(f"Fixing the query (attempt {attempt_no} of {self.max_retries + 1})…")
                gen = self.generator.repair(question, context, history, failed_sql, last_error or "")

            if gen.cannot_answer:
                response.attempts.append(Attempt(attempt_no, None, "generation", gen.cannot_answer_reason))
                response.status = "cannot_answer"
                response.error = gen.cannot_answer_reason
                return

            progress("Validating SQL…")
            check = self.validator.validate(gen.sql or "")
            if not check.is_valid:
                response.attempts.append(Attempt(attempt_no, gen.sql, "validation", check.error_message))
                response.sql = gen.sql
                if check.is_security_violation:
                    response.status = "blocked"
                    response.error = f"Query blocked by the safety layer: {check.error_message}"
                    return
                failed_sql, last_error = gen.sql, f"Validation error: {check.error_message}"
                if check.tables:
                    last_error += f"\nApproved tables are: {', '.join(self.validator.allowed_table_list)}."
                continue

            progress("Running query…")
            try:
                result = self.executor.execute(check.sql)
            except QueryExecutionError as exc:
                response.attempts.append(Attempt(attempt_no, check.sql, "execution", exc.message))
                response.sql = check.sql
                response.execution_ms = exc.execution_ms
                if not exc.retryable:
                    response.status = "blocked"
                    response.error = exc.message
                    return
                failed_sql, last_error = check.sql, f"PostgreSQL error: {exc.message}"
                if exc.is_timeout:
                    last_error += "\nRewrite the query to be more efficient (filter earlier, avoid cross joins)."
                continue

            response.attempts.append(Attempt(attempt_no, check.sql, "success"))
            response.status = "success"
            response.sql = check.sql
            response.limit_applied = check.limit_applied
            response.df = result.df
            response.row_count = result.row_count
            response.truncated = result.truncated
            response.execution_ms = result.execution_ms
            response.error = None
            return

        response.status = "error"
        response.error = (
            f"Could not produce a working query after {self.max_retries + 1} attempts. Last error: {last_error}"
        )


@dataclass
class _ResultView:
    """Adapter exposing an AgentResponse with the QueryResult attributes summaries need."""

    response: AgentResponse

    @property
    def df(self) -> pd.DataFrame:
        return self.response.df if self.response.df is not None else pd.DataFrame()

    @property
    def row_count(self) -> int:
        return self.response.row_count

    @property
    def truncated(self) -> bool:
        return self.response.truncated
