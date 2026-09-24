# Weather Agent

A small CLI agent that answers questions about the weather data from
[weather-app](https://github.com/Haroradk/weather-app) in plain English, by
writing and running its own SQL.

This is a different kind of learning project from that one: not pipeline
plumbing, but the fundamentals of an LLM **agent** - a model that can decide
to call a tool, see the result, and use it to answer, rather than just
completing text.

## The agent loop

1. You ask a question (e.g. *"was Copenhagen colder than usual last week?"*).
2. The model decides whether it needs data. If so, it returns a **function
   call** - structured JSON saying "run this SQL" - instead of a text answer.
3. We actually run that SQL (through a validator, see below) and hand the
   result back to the model as part of the conversation.
4. The model reads the result and either asks for more data or gives a final
   plain-English answer.

That loop - model, tool, repeat until done - is the core of what "an AI
agent" means in practice. It looks almost identical whether the model behind
it is Gemini, Claude, GPT, or a local model; only the exact request/response
shape changes.

## Why Gemini, and why this SQL validator

- **LLM**: Google's Gemini API free tier - a hosted API call, not a
  downloaded model, and a personal signup (your own Google account, no card,
  no org involvement).
- **Safety**: the agent's only tool that takes free-form SQL is
  `run_readonly_sql` (`sql_tool.py`). Its other tool, `describe_table`, only runs
  a fixed, parameterized catalog query. Every free-form query passes through a
  validator before it touches the database: only `SELECT`/`WITH` as the first keyword, a
  blocklist of dangerous keywords (`INSERT`, `DROP`, `ALTER`, ...) checked
  anywhere in the query, and no stacked (`;`-separated) statements.

**Be clear-eyed about what this is and isn't**: MotherDuck's read-only
tokens need a paid plan, which we don't have, so the token in `.env` is
read-write at the platform level. The validator above is the *only* thing
stopping a write - there's no database-enforced backstop behind it. That's
a real pattern used in production text-to-SQL tools when a DB-level
read-only role isn't available, but it's still "our code caught it" rather
than "the database refused it." If you ever upgrade MotherDuck plans, swap
in a real read-only token in `config.py` and you get a second, independent
layer for free.

## Setup

1. Get a free Gemini API key: go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey),
   sign in with your own Google account, click "Create API key".
2. Paste it into `.env` as `GEMINI_API_KEY=...` (already gitignored).
3. Run:

```bash
source .venv/bin/activate
python agent.py
```

Try asking things like:
- "What was the average temperature in London last week?"
- "Which city had the most rain in the last 30 days?"
- "How accurate has the forecast been for New York?"

Every tool call and its raw result print to the terminal as `[tool]` /
`[tool result]` lines - deliberately visible, not hidden, so you can see
exactly what SQL the model chose to write and whether it made sense.

### Chat UI (and testing it without spending Gemini quota)

```bash
source .venv/bin/activate
streamlit run app.py
```

Opens at http://localhost:8502. Same agent loop as the CLI (imports
`run_agent_turn` from `agent.py` directly, doesn't duplicate it), in a chat
interface - each tool call renders as an expandable bubble showing the SQL
and its result.

The free Gemini tier is tight (5 requests/minute, 20/day), so there's a
**"Demo mode"** checkbox in the sidebar for exercising the UI without
spending any of it: it skips the LLM entirely. Instead of typing a
question, you pick one from a fixed dropdown list, it runs the real query
behind that question against MotherDuck, and slots the actual returned
values into a canned sentence template (e.g. "Copenhagen had the most
rain, with 4.2mm on 2026-09-18"). The *data* is real; the *sentence* is
scripted, not reasoned - it's for checking the interface works, not for
asking it anything real. Turn it off to talk to the actual agent.

## Cloud deployment

Deployed at [haroradk-weatheragent.streamlit.app](https://haroradk-weatheragent.streamlit.app),
with `MOTHERDUCK_TOKEN` and `GEMINI_API_KEY` as Streamlit Cloud secrets (`app.py` bridges
`st.secrets` into environment variables, the same way weather-app does).

Caveat: the free Gemini quota (5 req/min, 20/day) is per API key, not per visitor. Every visitor
shares the one key in secrets, so a few questions from anyone can use up the day's quota for
everyone. For solo use that's fine, and Demo mode still works once the quota is gone. A public
version would need each visitor to paste in their own free key.

## Where the agent's knowledge of the data comes from

The agent has no hand-written schema. `catalog_tool.py` reads the warehouse's own catalog,
which [weather-app](https://github.com/Haroradk/weather-app)'s `semantic_layer.yml` publishes
into MotherDuck on every pipeline run: table/column descriptions as DuckDB `COMMENT`s, and
business metrics in `gold.metric_definitions`. Change a description or a metric definition there,
and the agent (and the dashboard) pick it up on the next pipeline run, with no code change here.

What goes where is a cost trade-off, because on the free tier every tool call is one more
Gemini request:
- **In the system prompt (free):** the list of tables with their descriptions, plus every metric
  with its exact SQL expression. It's small, and it's needed for almost every question.
- **As a tool (`describe_table`):** column-level detail, fetched only for tables a question
  actually touches.

Demo mode's first question ("most rainy days") builds its SQL from `gold.metric_definitions` at
run time, so you can watch the semantic layer drive a query without spending any quota.

## Searching what forecasters wrote (RAG)

`search_forecast_discussions` (`search_tool.py`) is the retrieval half of RAG
(retrieval-augmented generation). weather-app's pipeline stores National Weather Service forecaster
discussions (New York) split into sections, each with an embedding. This tool embeds the question
the same way and ranks sections by cosine similarity in DuckDB, then the model answers from those
passages. Matching is by meaning: "is it safe to swim at the beach?" finds the section about rip
currents, although it never uses the words "safe" or "swim".

The embedding model and vector size are read from the table rather than hard-coded, because a
query embedded with a different model gives meaningless rankings without raising any error. Each
search costs one embedding call, on a different model (and quota) from the chat model.

## What's deliberately simple here (and worth pushing on next)

- **No conversation memory across restarts.** `contents` lives in memory for
  one CLI session only.
- **A weak model on a small warehouse will sometimes write wrong SQL.**
  Watch the `[tool]` lines - if an answer looks off, the SQL it ran is
  printed right above it, so you can see exactly why.
