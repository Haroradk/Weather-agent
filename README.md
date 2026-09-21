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
- **Safety**: the agent gets exactly one tool, `run_readonly_sql`
  (`sql_tool.py`), and every query passes through a validator before it
  touches the database: only `SELECT`/`WITH` as the first keyword, a
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

## What's deliberately simple here (and worth pushing on next)

- **One tool, no schema discovery.** The model is handed the schema directly
  in the system prompt rather than having a `list_tables`/`describe_table`
  tool to explore with. Simpler for now; a `list_tables` tool would be a
  natural next step and a good exercise in tool design.
- **No conversation memory across restarts.** `contents` lives in memory for
  one CLI session only.
- **A weak model on a small warehouse will sometimes write wrong SQL.**
  Watch the `[tool]` lines - if an answer looks off, the SQL it ran is
  printed right above it, so you can see exactly why.
