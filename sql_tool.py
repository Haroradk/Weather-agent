"""
The one tool the agent gets: run a read-only SQL query and see the result.

Safety model (read this before trusting this against anything you actually
care about): the MotherDuck token in .env is read-write at the platform
level - read-only tokens need a paid plan, which we don't have. So the
*only* thing stopping the model from writing/dropping/altering data is
this validator. That's a real, legitimate pattern (plenty of production
text-to-SQL tools work exactly this way when a DB-level read-only role
isn't available) - but it's still "our code caught it", not "the database
refused it". If you ever upgrade MotherDuck plans, swap in a real
read-only token in config.py and you get a second, stronger layer for free.
"""

from __future__ import annotations

import re

import duckdb

FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "ATTACH", "DETACH",
    "COPY", "EXPORT", "IMPORT", "PRAGMA", "SET", "CALL", "GRANT", "REVOKE",
    "TRUNCATE", "MERGE", "REPLACE", "VACUUM", "CHECKPOINT",
]

MAX_ROWS_RETURNED = 50


def validate_readonly_sql(sql: str) -> str | None:
    """Returns an error message if the query isn't allowed, else None."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return "empty query"
    if ";" in stripped:
        return "multiple statements are not allowed - one SELECT per call"

    first_word = stripped.split(None, 1)[0].upper()
    if first_word not in ("SELECT", "WITH"):
        return f"only SELECT (or WITH ... SELECT) queries are allowed, got a query starting with {first_word}"

    upper = stripped.upper()
    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            return f"forbidden keyword detected: {keyword}"

    return None


def run_readonly_sql(con: duckdb.DuckDBPyConnection, sql: str) -> str:
    """Validates, runs, and formats a query as plain text for the model to read."""
    error = validate_readonly_sql(sql)
    if error:
        return f"QUERY REJECTED: {error}"

    try:
        con.execute(sql)
        rows = con.fetchall()
        columns = [d[0] for d in con.description]
    except Exception as e:
        return f"QUERY FAILED: {e}"

    if not rows:
        return "Query ran successfully but returned no rows."

    truncated = len(rows) > MAX_ROWS_RETURNED
    shown_rows = rows[:MAX_ROWS_RETURNED]

    lines = [" | ".join(columns)]
    for row in shown_rows:
        lines.append(" | ".join(str(value) for value in row))
    result = "\n".join(lines)

    if truncated:
        result += f"\n... ({len(rows) - MAX_ROWS_RETURNED} more row(s) truncated)"

    return result
