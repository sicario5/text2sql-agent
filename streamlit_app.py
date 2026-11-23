"""Streamlit frontend for the Text-to-SQL RAG agent.

Run with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
import time

import streamlit as st

from app.agent import AgentResponse
from app.config import Settings
from app.db import ConnectionStatus, check_connection, create_db_engine
from app.knowledge_base import add_verified_example, knowledge_base_counts
from app.llm import create_vanna
from app.results import format_value
from app.schema_inspector import inspect_schema, list_schemas
from app.services import AppServices, DatabaseUnavailableError, UnsafeDatabaseRoleError, build_services
from app.visualization import build_figure, compatible_chart_types, spec_for_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="Text-to-SQL Agent", page_icon="🗄️", layout="wide")

EXAMPLE_QUESTIONS = [
    "What were our total sales last month?",
    "Show the top 10 customers by revenue.",
    "Which products have the highest sales?",
    "Compare monthly revenue for the last year.",
    "Find customers who haven't placed an order in the last 90 days.",
    "What share of orders comes from each sales channel?",
]
STATUS_ICONS = {"success": "✅", "blocked": "🛡️", "cannot_answer": "🤔", "error": "❌"}


# --------------------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner=False)
def get_settings() -> Settings:
    return Settings.from_env()


@st.cache_resource(show_spinner=False)
def get_engine(database_url: str, timeout: int):
    return create_db_engine(database_url, timeout)


@st.cache_data(ttl=60, show_spinner=False)
def get_status(database_url: str, timeout: int) -> ConnectionStatus:
    return check_connection(get_engine(database_url, timeout))


@st.cache_data(ttl=300, show_spinner=False)
def get_schemas(database_url: str, timeout: int) -> list[str]:
    return list_schemas(get_engine(database_url, timeout))


@st.cache_data(ttl=300, show_spinner=False)
def get_schema_tables(database_url: str, timeout: int, schema: str, allowed: tuple[str, ...]) -> list[str]:
    info = inspect_schema(get_engine(database_url, timeout), schema, list(allowed) or None, sample_values=False)
    return info.table_names


@st.cache_resource(show_spinner=False)
def get_vanna(database: str, schema: str):
    return create_vanna(get_settings(), database, schema)


@st.cache_resource(show_spinner="Inspecting schema and loading the knowledge base…")
def get_services(schema: str, approved: tuple[str, ...]) -> AppServices:
    settings = get_settings()
    engine = get_engine(settings.database_url, settings.query_timeout_seconds)
    status = get_status(settings.database_url, settings.query_timeout_seconds)
    return build_services(
        settings,
        schema=schema,
        approved_tables=list(approved),
        engine=engine,
        vanna_instance=get_vanna(status.database, schema),
    )


# --------------------------------------------------------------------------- state
def init_state() -> None:
    st.session_state.setdefault("messages", [])  # [{"role": ..., "content": ..., "response": AgentResponse}]
    st.session_state.setdefault("pending_question", None)
    st.session_state.setdefault("active_schema", None)


def history_turns() -> list:
    return [
        m["response"].to_turn()
        for m in st.session_state.messages
        if m["role"] == "assistant" and m.get("response") is not None and m["response"].ok
    ]


# --------------------------------------------------------------------------- sidebar
def render_sidebar(settings: Settings) -> tuple[str, tuple[str, ...]] | None:
    with st.sidebar:
        st.header("🗄️ Text-to-SQL Agent")

        # --- connection status
        st.subheader("Database")
        status = get_status(settings.database_url, settings.query_timeout_seconds)
        if not status.ok:
            st.error(f"Not connected\n\n`{status.error}`")
            st.caption(settings.safe_database_url)
            if st.button("Retry connection", icon="🔄"):
                st.cache_data.clear()
                st.rerun()
            return None
        st.success(f"Connected to **{status.database}** as `{status.user}`", icon="🟢")
        st.caption(f"PostgreSQL {status.server_version.split()[0]}")
        if status.read_only:
            st.caption("🔒 Read-only role verified")
        else:
            st.warning("Role is not read-only:\n\n" + "\n".join(f"- {w}" for w in status.warnings))

        # --- schema and table selection
        schemas = get_schemas(settings.database_url, settings.query_timeout_seconds) or [settings.db_schema]
        default_idx = schemas.index(settings.db_schema) if settings.db_schema in schemas else 0
        schema = st.selectbox("Schema", schemas, index=default_idx)
        tables = get_schema_tables(
            settings.database_url, settings.query_timeout_seconds, schema, tuple(settings.allowed_tables)
        )
        approved = st.multiselect(
            "Approved tables",
            tables,
            default=tables,
            help="Queries may only touch these tables. The list can narrow, but never widen, ALLOWED_TABLES.",
        )
        if not approved:
            st.warning("Select at least one table.")
            return None

        # --- knowledge base
        st.subheader("Knowledge base")
        try:
            services = get_services(schema, tuple(sorted(approved)))
        except (DatabaseUnavailableError, UnsafeDatabaseRoleError, ValueError) as exc:
            st.error(str(exc))
            return None
        counts = knowledge_base_counts(services.vn)
        c1, c2, c3 = st.columns(3)
        c1.metric("DDL", counts.get("ddl", 0))
        c2.metric("Docs", counts.get("documentation", 0))
        c3.metric("Examples", counts.get("sql", 0))
        b1, b2 = st.columns(2)
        if b1.button("Train", icon="🧠", width="stretch", help="Add new schema items, docs and examples"):
            with st.spinner("Training…"):
                result = services.train()
            st.toast(f"Knowledge base updated: {result}")
            st.rerun()
        if b2.button("Reset", icon="♻️", width="stretch", help="Clear and rebuild from schema + knowledge files"):
            with st.spinner("Rebuilding…"):
                services.train(reset=True)
            st.toast("Knowledge base rebuilt")
            st.rerun()

        # --- settings
        with st.expander("Settings", icon="⚙️"):
            st.caption(f"LLM: `{settings.llm_label}`")
            st.caption(f"Embeddings: `{settings.embedding_provider}`")
            st.caption(
                f"Max rows: {settings.max_rows:,} · Timeout: {settings.query_timeout_seconds}s · "
                f"Retries: {settings.max_retries}"
            )
            services.agent.summary_enabled = st.toggle(
                "AI summary of results",
                value=settings.summary_enabled,
                help="Sends up to SUMMARY_MAX_ROWS result rows to the LLM to write a summary.",
            )

        if st.button("Clear chat", icon="🧹", width="stretch"):
            st.session_state.messages = []
            st.rerun()

        st.subheader("Try asking")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex_{q}", type="tertiary"):
                st.session_state.pending_question = q
    return schema, tuple(sorted(approved))


# --------------------------------------------------------------------------- rendering
def render_response(resp: AgentResponse, idx: int, services: AppServices) -> None:
    if resp.standalone_question and resp.standalone_question != resp.question:
        st.caption(f"🔁 Interpreted as: *{resp.standalone_question}*")

    if resp.status == "success":
        st.markdown(resp.summary or "Query completed.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{resp.row_count:,}" + ("+" if resp.truncated else ""))
        m2.metric("Execution time", f"{resp.execution_ms:,.0f} ms")
        m3.metric("Total time", f"{resp.total_ms / 1000:,.1f} s")
        m4.metric("Attempts", len(resp.attempts))
        if resp.truncated:
            st.warning(f"Showing the first {resp.row_count:,} rows (row limit reached).", icon="✂️")
    elif resp.status == "blocked":
        st.error(resp.error, icon="🛡️")
    elif resp.status == "cannot_answer":
        st.info(f"I can't answer that from this database: {resp.error}", icon="🤔")
    else:
        st.error(resp.error or "Something went wrong.", icon="❌")

    if resp.sql:
        with st.expander("Generated SQL", icon="🧾", expanded=resp.status != "success"):
            st.code(resp.sql, language="sql")
            if resp.limit_applied:
                st.caption("A row LIMIT was added by the safety layer.")
            if resp.context_counts:
                c = resp.context_counts
                st.caption(
                    f"RAG context: {c.get('ddl', 0)} tables · {c.get('documentation', 0)} docs · "
                    f"{c.get('examples', 0)} examples"
                )

    failed = [a for a in resp.attempts if a.stage != "success"]
    if failed and (len(resp.attempts) > 1 or resp.status == "error"):
        with st.expander(f"Self-correction log ({len(failed)} failed attempt(s))", icon="🔁"):
            for a in failed:
                st.markdown(f"**Attempt {a.number}** — {a.stage} error: {a.error}")
                if a.sql:
                    st.code(a.sql, language="sql")

    if resp.status != "success" or resp.df is None:
        return

    df = resp.df
    chart_tab, table_tab = st.tabs(["📊 Chart", "📋 Table"])
    with chart_tab:
        options = compatible_chart_types(df)
        recommended = resp.chart.chart_type if resp.chart and resp.chart.chart_type in options else "table"
        chosen = st.selectbox(
            "Chart type",
            options,
            index=options.index(recommended),
            key=f"chart_type_{idx}",
            help=resp.chart.reason if resp.chart else None,
            label_visibility="collapsed",
        )
        spec = spec_for_type(df, chosen, resp.chart)
        if spec.chart_type == "kpi":
            cols = st.columns(max(1, len(spec.y)))
            for col, name in zip(cols, spec.y, strict=False):
                col.metric(name.replace("_", " ").title(), format_value(df.iloc[0][name]), border=True)
        elif spec.chart_type == "table":
            st.dataframe(df, hide_index=True, height=min(400, 38 + 35 * len(df)))
        else:
            fig = build_figure(df, spec)
            if fig is None:
                st.info("This chart type doesn't fit the result; showing the table.")
                st.dataframe(df, hide_index=True)
            else:
                st.plotly_chart(fig, key=f"fig_{idx}_{chosen}")
    with table_tab:
        st.dataframe(df, hide_index=True)

    a1, a2, _ = st.columns([1, 1, 3])
    a1.download_button(
        "Export CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"query_result_{idx}.csv",
        mime="text/csv",
        icon="⬇️",
        key=f"csv_{idx}",
        on_click="ignore",
    )
    if a2.button("Save as example", icon="👍", key=f"save_{idx}", help="Teach the agent: store this question + SQL"):
        try:
            add_verified_example(services.vn, services.validator, resp.standalone_question, resp.sql or "")
            st.toast("Saved to the knowledge base. Similar questions will reuse this SQL.", icon="🧠")
        except ValueError as exc:
            st.toast(str(exc), icon="⚠️")


def run_question(question: str, services: AppServices) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        started = time.perf_counter()
        with st.status("Working on it…", expanded=False) as status_box:

            def progress(msg: str) -> None:
                status_box.update(label=msg)
                status_box.write(msg)

            response = services.agent.ask(question, history_turns(), on_progress=progress)
            elapsed = time.perf_counter() - started
            state = "complete" if response.ok else "error"
            status_box.update(label=f"{STATUS_ICONS.get(response.status, '')} Finished in {elapsed:.1f}s", state=state)
        idx = len(st.session_state.messages)
        render_response(response, idx, services)
    st.session_state.messages.append({"role": "assistant", "content": response.summary, "response": response})


# --------------------------------------------------------------------------- main
def main() -> None:
    init_state()
    try:
        settings = get_settings()
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        st.info("Copy `.env.example` to `.env` and fill in the database and LLM settings.")
        st.stop()

    selection = render_sidebar(settings)
    st.title("Ask your database")
    st.caption("Natural-language questions → validated, read-only SQL → answers and charts.")
    if selection is None:
        st.stop()
    schema, approved = selection

    if st.session_state.active_schema != schema:
        st.session_state.active_schema = schema
        st.session_state.messages = []

    services = get_services(schema, approved)

    if not st.session_state.messages:
        st.markdown("#### Try one of these")
        cols = st.columns(3)
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 3].button(q, key=f"hero_{i}", width="stretch"):
                st.session_state.pending_question = q

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_response(message["response"], i, services)

    typed = st.chat_input("Ask a question about your data…")
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None
    if question:
        run_question(question, services)


main()
