"""SQL validation and safety layer.

Every SQL statement produced by the LLM passes through ``SQLValidator.validate``
before it can reach the database. The validator parses the SQL into an AST with
sqlglot (PostgreSQL dialect) and enforces:

1. exactly one statement (no ``SELECT 1; DROP TABLE x`` stacking);
2. the statement is a read-only query: SELECT / WITH ... SELECT / UNION etc.;
3. no data-modifying or DDL nodes anywhere in the tree, including inside CTEs
   (``WITH d AS (DELETE ... RETURNING *) SELECT ...``), ``SELECT ... INTO`` and
   row locks (``FOR UPDATE``);
4. no dangerous functions (``pg_sleep``, ``pg_read_file``, ``dblink``,
   ``query_to_xml`` which executes arbitrary SQL strings, ``nextval`` / ``setval``
   which mutate sequences, ``set_config`` ...);
5. every referenced table is on the approved list; system catalogs and other
   schemas are rejected;
6. a row LIMIT no larger than ``max_rows``.

The SQL that is executed is re-generated from the validated AST (comments
stripped), so what runs is exactly what was checked.

This layer is defence in depth. The database role is read-only, every
transaction is opened READ ONLY, and statements run under a timeout, so even a
validator bug cannot modify data.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError
from sqlglot.tokens import TokenType

DIALECT = "postgres"
MAX_SQL_LENGTH = 20_000

# Keywords that never belong in a read-only analytical query. Checked on the token
# stream, so the words are allowed inside string literals and quoted identifiers.
FORBIDDEN_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "UPSERT",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "COPY",
    "CALL",
    "EXECUTE",
    "EXEC",
    "DO",
    "VACUUM",
    "REINDEX",
    "CLUSTER",
    "LOCK",
    "PREPARE",
    "DEALLOCATE",
    "DISCARD",
    "LISTEN",
    "NOTIFY",
    "UNLISTEN",
    "REFRESH",
    "SET",
    "RESET",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "INTO",
}

# AST node types that must never appear anywhere in the tree.
_FORBIDDEN_NODE_NAMES = (
    "Insert",
    "Update",
    "Delete",
    "Merge",
    "Drop",
    "Create",
    "Alter",
    "AlterTable",
    "TruncateTable",
    "Command",
    "Copy",
    "Set",
    "Transaction",
    "Commit",
    "Rollback",
    "Into",
    "Lock",
    "Grant",
    "Revoke",
    "Use",
    "Pragma",
    "LoadData",
    "Analyze",
)
FORBIDDEN_NODE_TYPES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))

ALLOWED_ROOT_TYPES = (exp.Select, exp.SetOperation, exp.Subquery)

# Functions that can sleep, read files, open connections, execute dynamic SQL,
# change settings, signal backends or mutate sequences.
BLOCKED_FUNCTIONS = {
    "set_config",
    "current_setting",
    "nextval",
    "setval",
    "currval",
    "lastval",
    "query_to_xml",
    "query_to_xml_and_xmlschema",
    "query_to_xmlschema",
    "table_to_xml",
    "table_to_xml_and_xmlschema",
    "table_to_xmlschema",
    "cursor_to_xml",
    "cursor_to_xmlschema",
    "schema_to_xml",
    "schema_to_xml_and_xmlschema",
    "schema_to_xmlschema",
    "database_to_xml",
    "database_to_xml_and_xmlschema",
    "database_to_xmlschema",
    "txid_current",
    "txid_current_if_assigned",
    "has_table_privilege",
    "has_schema_privilege",
    "has_column_privilege",
    "has_database_privilege",
    "has_function_privilege",
    "version",
    "inet_server_addr",
    "inet_server_port",
    "inet_client_addr",
    "current_user",
    "session_user",
    "current_database",
    "current_schema",
    "current_schemas",
    "make_lsn",
}
BLOCKED_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "xpath", "binary_upgrade_", "brin_", "gin_")

_STRING_TOKENS = tuple(
    getattr(TokenType, n)
    for n in ("STRING", "NATIONAL_STRING", "HEREDOC_STRING", "RAW_STRING", "BIT_STRING", "HEX_STRING", "BYTE_STRING")
    if hasattr(TokenType, n)
)

BLOCKED_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast"}


class ErrorCode(StrEnum):
    EMPTY = "empty"
    TOO_LONG = "too_long"
    PARSE_ERROR = "parse_error"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NOT_SELECT = "not_select"
    FORBIDDEN_KEYWORD = "forbidden_keyword"
    FORBIDDEN_OPERATION = "forbidden_operation"
    FORBIDDEN_FUNCTION = "forbidden_function"
    DISALLOWED_TABLE = "disallowed_table"
    SYSTEM_CATALOG = "system_catalog"


# Errors the LLM cannot fix by rewriting the query: the request itself is unsafe.
SECURITY_ERROR_CODES = {
    ErrorCode.NOT_SELECT,
    ErrorCode.FORBIDDEN_KEYWORD,
    ErrorCode.FORBIDDEN_OPERATION,
    ErrorCode.FORBIDDEN_FUNCTION,
    ErrorCode.SYSTEM_CATALOG,
    ErrorCode.MULTIPLE_STATEMENTS,
}


@dataclass
class ValidationIssue:
    code: ErrorCode
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass
class ValidationResult:
    is_valid: bool
    original_sql: str
    sql: str | None = None  # safe, normalised SQL to execute (only when valid)
    issues: list[ValidationIssue] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    limit_applied: bool = False

    @property
    def errors(self) -> list[str]:
        return [i.message for i in self.issues]

    @property
    def is_security_violation(self) -> bool:
        return any(i.code in SECURITY_ERROR_CODES for i in self.issues)

    @property
    def error_message(self) -> str:
        return "; ".join(self.errors)


class SQLValidator:
    def __init__(
        self,
        allowed_tables: Iterable[str],
        default_schema: str = "public",
        max_rows: int = 1000,
        blocked_functions: Iterable[str] | None = None,
    ) -> None:
        self.default_schema = default_schema.lower()
        self.max_rows = max_rows
        self.allowed_tables = {self._normalise_table_name(t) for t in allowed_tables}
        self.blocked_functions = set(BLOCKED_FUNCTIONS if blocked_functions is None else blocked_functions)

    # ------------------------------------------------------------------ public
    def validate(self, sql: str) -> ValidationResult:
        result = ValidationResult(is_valid=False, original_sql=sql or "")
        cleaned = (sql or "").strip()

        if not cleaned:
            return self._fail(result, ErrorCode.EMPTY, "The SQL query is empty.")
        if len(cleaned) > MAX_SQL_LENGTH:
            return self._fail(result, ErrorCode.TOO_LONG, f"The SQL query exceeds {MAX_SQL_LENGTH} characters.")
        if "\x00" in cleaned:
            return self._fail(result, ErrorCode.PARSE_ERROR, "The SQL query contains a NUL byte.")

        # 1) token-level keyword screen (comments and string literals are not tokens)
        try:
            tokens = sqlglot.tokenize(cleaned, read=DIALECT)
        except TokenError as exc:
            return self._fail(result, ErrorCode.PARSE_ERROR, f"Could not tokenize SQL: {exc}")
        for token in tokens:
            if token.token_type in _STRING_TOKENS:
                continue
            if token.token_type == TokenType.IDENTIFIER:  # "quoted" identifier
                continue
            word = token.text.upper()
            if word in FORBIDDEN_KEYWORDS:
                return self._fail(
                    result,
                    ErrorCode.FORBIDDEN_KEYWORD,
                    f"Forbidden keyword '{word}': only read-only SELECT queries are allowed.",
                )

        # 2) parse into an AST
        try:
            statements = [s for s in sqlglot.parse(cleaned, read=DIALECT) if s is not None]
        except (ParseError, TokenError) as exc:
            return self._fail(result, ErrorCode.PARSE_ERROR, f"SQL syntax error: {_first_line(exc)}")
        if not statements:
            return self._fail(result, ErrorCode.EMPTY, "No SQL statement found.")
        if len(statements) > 1:
            return self._fail(
                result, ErrorCode.MULTIPLE_STATEMENTS, "Multiple SQL statements are not allowed; send one query."
            )
        tree = statements[0]

        # 3) statement type
        if not isinstance(tree, ALLOWED_ROOT_TYPES) or isinstance(tree, exp.Command):
            return self._fail(
                result,
                ErrorCode.NOT_SELECT,
                f"Only SELECT queries are allowed (got {tree.key.upper()}).",
            )

        # 4) forbidden operations anywhere in the tree
        for node in tree.walk():
            if isinstance(node, FORBIDDEN_NODE_TYPES):
                label = {"Into": "SELECT ... INTO", "Lock": "row locking (FOR UPDATE/SHARE)"}.get(
                    type(node).__name__, type(node).__name__.upper()
                )
                return self._fail(
                    result, ErrorCode.FORBIDDEN_OPERATION, f"Forbidden operation: {label} is not allowed."
                )

        # 5) functions (check every spelling: sqlglot's internal name and the rendered Postgres name)
        for func in tree.find_all(exp.Func):
            for name in self._function_names(func):
                if self._is_blocked_function(name):
                    return self._fail(result, ErrorCode.FORBIDDEN_FUNCTION, f"Function '{name}' is not allowed.")

        # 6) tables
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        referenced: set[str] = set()
        for table in tree.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                continue  # table-valued function such as generate_series(); checked as a function above
            name = self._identifier(table.this)
            if not name:
                continue
            schema = self._identifier(table.args.get("db")) if table.args.get("db") else None
            catalog = table.args.get("catalog")
            if catalog is not None:
                result.issues.append(
                    ValidationIssue(
                        ErrorCode.DISALLOWED_TABLE, f"Cross-database reference '{table.sql()}' is not allowed."
                    )
                )
                continue
            if schema is None and name in cte_names:
                continue  # reference to a CTE, not a real table
            if schema in BLOCKED_SCHEMAS or (schema is None and name.startswith("pg_")):
                return self._fail(
                    result, ErrorCode.SYSTEM_CATALOG, f"Access to system catalog '{table.sql()}' is not allowed."
                )
            qualified = f"{schema or self.default_schema}.{name}"
            if qualified not in self.allowed_tables:
                result.issues.append(
                    ValidationIssue(
                        ErrorCode.DISALLOWED_TABLE,
                        f"Table '{qualified}' does not exist or is not approved for querying.",
                    )
                )
            referenced.add(qualified)
        result.tables = sorted(referenced)
        if result.issues:
            return result

        # 7) row limit
        tree, limit_applied = self._enforce_limit(tree)
        result.limit_applied = limit_applied
        result.sql = tree.sql(dialect=DIALECT, pretty=True, comments=False)
        result.is_valid = True
        return result

    @property
    def allowed_table_list(self) -> list[str]:
        return sorted(self.allowed_tables)

    # ----------------------------------------------------------------- helpers
    def _normalise_table_name(self, name: str) -> str:
        name = name.strip().strip('"').lower()
        return name if "." in name else f"{self.default_schema}.{name}"

    @staticmethod
    def _identifier(node) -> str | None:
        if node is None:
            return None
        if isinstance(node, exp.Identifier):
            # Postgres folds unquoted identifiers to lower case; quoted ones keep their case.
            return node.this if node.quoted else node.this.lower()
        return str(node.name if hasattr(node, "name") else node).lower()

    @staticmethod
    def _function_names(func: exp.Func) -> set[str]:
        if isinstance(func, exp.Anonymous):
            return {str(func.name).lower()}
        names = {func.sql_name().lower()}
        try:
            rendered = func.sql(dialect=DIALECT)
            names.add(re.split(r"[\s(]", rendered, maxsplit=1)[0].lower())
        except Exception:  # noqa: BLE001 - rendering is best effort
            pass
        return {n for n in names if n}

    def _is_blocked_function(self, name: str) -> bool:
        name = name.lower()
        return name in self.blocked_functions or name.startswith(BLOCKED_FUNCTION_PREFIXES)

    def _enforce_limit(self, tree: exp.Expression) -> tuple[exp.Expression, bool]:
        """Ensure the outermost query returns at most ``max_rows`` rows."""
        target = tree
        if isinstance(tree, exp.Subquery):
            target = tree.unnest()

        fetch = target.args.get("fetch")
        if fetch is not None:
            count = _literal_int(fetch.args.get("count"))
            if count is not None and count <= self.max_rows:
                return tree, False
            target.set("fetch", None)

        limit = target.args.get("limit")
        if limit is not None:
            # sqlglot stores both LIMIT n and FETCH FIRST n ROWS ONLY under "limit"
            count_node = limit.args.get("count") if isinstance(limit, exp.Fetch) else limit.expression
            count = _literal_int(count_node)
            if count is not None and count <= self.max_rows:
                return tree, False
        if isinstance(target, (exp.Select, exp.SetOperation)):
            target.set("limit", exp.Limit(expression=exp.Literal.number(self.max_rows)))
            return tree, True
        # Fallback: wrap the query (should not happen for allowed root types)
        wrapped = exp.select("*").from_(exp.Subquery(this=tree, alias=exp.TableAlias(this=exp.to_identifier("_q"))))
        return wrapped.limit(self.max_rows), True

    @staticmethod
    def _fail(result: ValidationResult, code: ErrorCode, message: str) -> ValidationResult:
        result.issues.append(ValidationIssue(code, message))
        result.is_valid = False
        return result


def _literal_int(node) -> int | None:
    if isinstance(node, exp.Literal) and not node.is_string:
        try:
            return int(node.this)
        except ValueError:
            return None
    return None


def _first_line(exc: Exception) -> str:
    text = re.sub(r"\x1b\[[0-9;]*m", "", str(exc))  # strip ANSI colours from sqlglot errors
    return text.strip().splitlines()[0][:300] if text.strip() else exc.__class__.__name__
