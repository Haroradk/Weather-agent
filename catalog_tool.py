"""
Reads the warehouse's own catalog (DuckDB COMMENTs + gold.metric_definitions,
published by weather-etl-pipeline's semantic_layer.yml) instead of the agent
carrying a hand-written copy of the schema that can drift out of date.

Two pieces, split by cost:
- build_catalog_summary(): table list + descriptions + metric definitions.
  Small, so it goes straight into the system prompt - zero extra LLM calls.
- describe_table(): column-level detail for one table. Exposed as a tool,
  so the model only pays a round trip for tables it actually needs.
"""

from __future__ import annotations

import duckdb

CATALOG_SCHEMA = "gold"

TABLES_QUERY = """
SELECT schema_name || '.' || table_name, comment FROM duckdb_tables()
WHERE database_name = current_database() AND schema_name = ?
UNION ALL
SELECT schema_name || '.' || view_name, comment FROM duckdb_views()
WHERE database_name = current_database() AND schema_name = ?
ORDER BY 1
"""

METRICS_QUERY = """
SELECT name, label, description, table_name, expression, filter, unit
FROM gold.metric_definitions ORDER BY name
"""

COLUMNS_QUERY = """
SELECT column_name, data_type, comment FROM duckdb_columns()
WHERE database_name = current_database() AND schema_name = ? AND table_name = ?
ORDER BY column_index
"""


def build_catalog_summary(con: duckdb.DuckDBPyConnection) -> str:
    lines = ["Tables and views:"]
    for name, comment in con.execute(TABLES_QUERY, [CATALOG_SCHEMA, CATALOG_SCHEMA]).fetchall():
        lines.append(f"- {name}: {comment or '(no description)'}")

    lines.append("")
    lines.append(
        "Business metrics - when a question matches one, use its exact expression and filter "
        "rather than inventing your own definition:"
    )
    for name, label, description, table, expression, filter_sql, unit in con.execute(METRICS_QUERY).fetchall():
        lines.append(
            f"- {name} ({label}, {unit}): {description} "
            f"Compute as {expression} FROM {table} WHERE {filter_sql}"
        )
    return "\n".join(lines)


def describe_table(con: duckdb.DuckDBPyConnection, table_name: str) -> str:
    parts = table_name.strip().split(".")
    if len(parts) == 1:
        schema, name = CATALOG_SCHEMA, parts[0]
    elif len(parts) == 2:
        schema, name = parts
    else:
        return f"Invalid table name {table_name!r} - use schema.table, e.g. gold.weather_daily_summary."

    rows = con.execute(COLUMNS_QUERY, [schema, name]).fetchall()
    if not rows:
        return f"No table or view named {schema}.{name}. See the table list in your instructions."

    lines = [f"{schema}.{name} columns:"]
    for column, data_type, comment in rows:
        lines.append(f"- {column} {data_type}: {comment or '(no description)'}")
    return "\n".join(lines)
