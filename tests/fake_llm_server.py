"""A fake OpenAI-compatible chat server for offline end-to-end tests.

It is NOT an AI. It answers like a model that copies the closest few-shot
example found in the prompt, which is enough to exercise the full stack
(OpenAI adapter -> RAG prompt -> validator -> PostgreSQL -> summary -> UI)
without an API key. It also scripts a few behaviours for the tests:

* "highest sales"   -> first answer uses a wrong column, fixed on the repair prompt
* "delete" / "drop" -> answers with a DELETE statement (must be blocked)
* "salary"          -> queries a forbidden table, then CANNOT_ANSWER on repair
* follow-ups "Now only for <year>" / "Compare it with <year>" are rewritten

Run standalone:  python tests/fake_llm_server.py --port 8765
Then set LLM_PROVIDER=openai_compatible, LLM_BASE_URL=http://localhost:8765/v1,
LLM_API_KEY=fake, LLM_MODEL=fake.
"""

from __future__ import annotations

import argparse
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EXAMPLE_RE = re.compile(r"Question: (.*?)\n```sql\n(.*?)\n```", re.DOTALL)
WORD_RE = re.compile(r"[a-z0-9]+")
STOP = {"the", "a", "an", "of", "for", "by", "in", "on", "what", "which", "show", "me", "is", "are", "our", "each"}


def _words(text: str) -> set[str]:
    return {w.rstrip("s") for w in WORD_RE.findall(text.lower()) if w not in STOP}


def _best_example(question: str, system: str) -> str | None:
    examples = EXAMPLE_RE.findall(system)
    if not examples:
        return None
    q = _words(question)
    scored = sorted(examples, key=lambda ex: len(q & _words(ex[0])) / (len(q | _words(ex[0])) or 1), reverse=True)
    return scored[0][1]


MONTHLY_FOR_YEAR = """SELECT date_trunc('month', o.order_date)::date AS month,
       ROUND(SUM(oi.quantity * oi.unit_price * (1 - oi.discount)), 2) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.status NOT IN ('cancelled', 'returned')
  AND o.order_date >= DATE '{y}-01-01' AND o.order_date < DATE '{y1}-01-01'
GROUP BY 1
ORDER BY 1"""


def respond(messages: list[dict]) -> str:
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    last = messages[-1]["content"]

    # --- follow-up rewriting
    if "You rewrite follow-up questions" in system:
        latest = last.split("Latest question:", 1)[1].strip()
        history_qs = re.findall(r"\(interpreted as: (.*?)\)|^\d+\. User: (.*)$", last, re.MULTILINE)
        previous = [a or b for a, b in history_qs][-1] if history_qs else ""
        if m := re.search(r"only for (\d{4})", latest, re.IGNORECASE):
            return f"{previous.rstrip('.')} for {m.group(1)}."
        if m := re.search(r"compare it with (\d{4})", latest, re.IGNORECASE):
            prev_year = re.search(r"(\d{4})", previous)
            return f"Compare monthly revenue in {prev_year.group(1) if prev_year else '2025'} with {m.group(1)}."
        return latest

    # --- summaries
    if "precise data analyst" in system:
        rows = last.split("```csv\n", 1)[1].split("\n```", 1)[0].strip().splitlines()
        header, first = rows[0], rows[1] if len(rows) > 1 else ""
        return f"(fake LLM summary) {len(rows) - 1} row(s) shown. Columns: {header}. First row: {first}."

    # --- SQL repair
    if last.startswith("That query failed"):
        question = re.search(r'original question: "(.*?)"', last, re.DOTALL).group(1)
        if "salar" in question.lower():
            return "CANNOT_ANSWER: there is no salary data in the approved tables."
        sql = _best_example(question, system) or "SELECT 1"
        return f"```sql\n{sql}\n```"

    # --- SQL generation
    question = last
    lower = question.lower()
    if "delete" in lower or "drop" in lower:
        return "```sql\nDELETE FROM orders\n```"
    if "salar" in lower:
        return "```sql\nSELECT full_name, salary_usd FROM internal.employee_salaries\n```"
    if m := re.search(r"by month for (\d{4})", lower):
        y = int(m.group(1))
        return f"```sql\n{MONTHLY_FOR_YEAR.format(y=y, y1=y + 1)}\n```"
    sql = _best_example(question, system)
    if sql is None:
        return "CANNOT_ANSWER: no similar example found."
    if "highest sales" in lower:  # deliberately wrong first attempt -> exercises self-correction
        sql = sql.replace("p.name AS product_name", "p.product_name AS product_name")
    return f"```sql\n{sql}\n```"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        content = respond(body.get("messages", []))
        payload = {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "fake"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence
        pass


def start_server(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    srv, url = start_server(args.port)
    print(f"Fake LLM listening on {url}  (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
