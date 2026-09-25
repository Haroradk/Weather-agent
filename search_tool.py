"""
Semantic search over National Weather Service forecasters' text - the
retrieval half of RAG (retrieval-augmented generation).

The pipeline (weather-app) already split each discussion into sections and
stored an embedding per section in silver.forecast_discussion_sections.
Here the question gets embedded the same way, and DuckDB's
array_cosine_similarity ranks sections by how close their meaning is - so
"will it be stormy at the beach?" can find a section that only ever says
"high surf" and "rip currents". The model then answers from those
passages, rather than from memory.

The embedding model and vector size are read from the table itself rather
than hard-coded, so this can't silently drift from what the pipeline used -
a query embedded with a different model would return nonsense rankings
without any error.
"""

from __future__ import annotations

import duckdb
from google.genai import types

SECTIONS_TABLE = "silver.forecast_discussion_sections"
TOP_K = 4
MAX_CHARS_PER_SECTION = 900


def find_forecast_sections(con: duckdb.DuckDBPyConnection, client, query: str, embed_with_retry=None) -> tuple[list[dict], str | None]:
    """Returns (matching sections, best first; a message instead if the search couldn't run)."""
    stored = con.execute(f"SELECT DISTINCT embedding_model, array_length(embedding) FROM {SECTIONS_TABLE}").fetchall()
    if not stored:
        return [], "No forecaster discussions have been indexed yet."
    if len(stored) > 1:
        return [], f"Index contains mixed embedding models {stored} - rankings would be meaningless; re-index first."
    model, dimensions = stored[0]

    def embed():
        return client.models.embed_content(
            model=model,
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=dimensions),
        ).embeddings[0].values

    vector = embed_with_retry(embed)() if embed_with_retry else embed()
    rows = con.execute(
        f"""
        SELECT issued_at, section_name, content,
               array_cosine_similarity(embedding, ?::FLOAT[{int(dimensions)}]) AS score
        FROM {SECTIONS_TABLE}
        ORDER BY score DESC
        LIMIT ?
        """,
        [vector, TOP_K],
    ).fetchall()
    sections = [
        {"issued_at": issued_at, "section": section, "content": content, "score": score}
        for issued_at, section, content, score in rows
    ]
    return sections, None


def format_sections(query: str, sections: list[dict]) -> str:
    """The plain text the model reads."""
    lines = [f"Top {len(sections)} forecaster-discussion sections for {query!r} (NWS New York office):"]
    for s in sections:
        content = s["content"]
        text = content if len(content) <= MAX_CHARS_PER_SECTION else content[:MAX_CHARS_PER_SECTION] + "..."
        lines.append(f"\n[{s['section']}, issued {s['issued_at']:%Y-%m-%d %H:%M} UTC, similarity {s['score']:.2f}]\n{text}")
    return "\n".join(lines)

