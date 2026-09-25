"""
A small CLI agent that answers weather questions by writing its own SQL.

The loop, in plain terms:
  1. You ask a question in English.
  2. The model (Gemini) decides whether it needs data. If so, it emits a
     "function call" - not real code, just structured JSON saying "call
     run_readonly_sql with this query" - instead of a text answer.
  3. We actually run that query (through the validator in sql_tool.py) and
     feed the result back to the model as a new turn.
  4. The model reads the result and either asks for more data, or writes
     a final plain-English answer. We print that and wait for your next
     question.

This is the same fundamental pattern behind every "agent" - a model, a
tool it can ask to be run on its behalf, and a loop that keeps handing
control back and forth until the model is done. Nothing here is
Gemini-specific in spirit; the same loop looks almost identical wired up
to Claude, GPT, or a local model - only the exact request/response shape
differs.
"""

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import config
from catalog_tool import build_catalog_summary, describe_table
from search_tool import find_forecast_sections, format_sections
from sql_tool import run_readonly_query


def _is_retryable(exception: BaseException) -> bool:
    """503 (transient overload) and 429 (rate limit - the free tier caps
    this model at 5 requests/minute, and the API's own error message says
    'please retry in Ns') both warrant a retry with backoff. A 400/403
    (bad request, bad API key) won't fix itself by retrying."""
    if isinstance(exception, genai_errors.ServerError):
        return True
    if isinstance(exception, genai_errors.ClientError):
        return exception.code == 429
    return False


# Free-tier rate limits are the main thing this needs to ride out - a
# 5-req/min cap means a wait of up to ~60s between attempts is realistic,
# not excessive.
def _log_retry(retry_state) -> None:
    print(f"  [retrying after {retry_state.outcome.exception()!r}, attempt {retry_state.attempt_number}]")


retry_on_transient_error = retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=3, max=60),
    before_sleep=_log_retry,
    reraise=True,
)

SYSTEM_PROMPT_TEMPLATE = """\
You are a weather-data assistant. You answer questions using ONLY real data
from the weather warehouse (DuckDB/MotherDuck SQL) - never guess or make up
numbers. If a question can't be answered from these tables, say so.

This catalog is read live from the warehouse's own descriptions. Read the
table descriptions carefully - they contain rules (e.g. which rows are
settled history vs. forecast) that change what the right query is.

{catalog}

Tools:
- describe_table: column names, types and descriptions for one table. Call
  it before querying a table whose columns you haven't seen yet.
- run_readonly_sql: run one SELECT query. Anything else is rejected.
- search_forecast_discussions: semantic search over what National Weather
  Service forecasters (New York office) wrote in their forecast discussions.
  Use it for questions about forecasters' reasoning, warnings, storms or
  hazards - things numbers alone don't capture. Quote what you use, with
  its issue date. New York only.

When you have your answer, respond in plain, concise English and cite the
specific numbers you found (don't just say "it was warmer", say by how much).
"""

TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="run_readonly_sql",
            description="Run a single read-only SELECT query against the weather warehouse and return the results as text.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "sql": types.Schema(
                        type="STRING",
                        description="A single SELECT (or WITH ... SELECT) statement. No INSERT/UPDATE/DELETE/DDL - it will be rejected.",
                    ),
                },
                required=["sql"],
            ),
        ),
        types.FunctionDeclaration(
            name="search_forecast_discussions",
            description="Find the passages of National Weather Service forecaster discussions (New York) most relevant to a question, by meaning rather than keywords.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="What to look for, in plain English, e.g. 'coastal flooding this weekend'.",
                    ),
                },
                required=["query"],
            ),
        ),
        types.FunctionDeclaration(
            name="describe_table",
            description="Get the columns of one warehouse table or view, with their types and descriptions.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "table_name": types.Schema(
                        type="STRING",
                        description="Fully qualified name, e.g. gold.weather_daily_summary.",
                    ),
                },
                required=["table_name"],
            ),
        ),
    ]
)


def _execute_tool(con, call, client) -> dict:
    """Runs one tool call. Returns a step: {"tool", "label", "result"} plus
    "df" (SQL result rows) or "sources" (matched passages) for a UI to show.
    "result" is the text the model reads."""
    if call.name == "run_readonly_sql":
        sql = call.args["sql"]
        result, df = run_readonly_query(con, sql)
        return {"tool": call.name, "label": sql, "arg": sql, "result": result, "df": df}
    if call.name == "describe_table":
        table_name = call.args["table_name"]
        return {
            "tool": call.name,
            "label": f"describe_table({table_name})",
            "arg": table_name,
            "result": describe_table(con, table_name),
        }
    if call.name == "search_forecast_discussions":
        query = call.args["query"]
        sections, message = find_forecast_sections(con, client, query, embed_with_retry=retry_on_transient_error)
        return {
            "tool": call.name,
            "label": f"search_forecast_discussions({query!r})",
            "arg": query,
            "result": message or format_sections(query, sections),
            "sources": sections,
        }
    return {"tool": call.name, "label": f"<unknown tool: {call.name}>", "result": f"ERROR: unknown tool {call.name}"}


def _print_step(step: dict) -> None:
    result = step["result"]
    print(f"  [tool] {step['label']}")
    print(f"  [tool result] {result[:300]}{'...' if len(result) > 300 else ''}")


def run_agent_turn(client: genai.Client, con, contents: list, on_step=None) -> str:
    """Runs the tool-use loop for one user turn, mutating `contents` in place
    with everything that happened, and returning the final text answer.

    on_step(step), if given, is called with each step as it happens (see
    _execute_tool; the first is the catalog read) instead of printing - lets
    a UI (e.g. app.py) show the agent's work live, without duplicating this loop."""
    catalog = build_catalog_summary(con)
    if on_step:
        tables_part, metrics_part = catalog.split("\n\n", 1)
        n_tables, n_metrics = (sum(line.startswith("- ") for line in part.splitlines()) for part in (tables_part, metrics_part))
        label = f"Read the warehouse catalog: {n_tables} tables, {n_metrics} metric definitions"
        on_step({"tool": "catalog", "label": label, "result": catalog})
    generate_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT_TEMPLATE.format(catalog=catalog),
        tools=[TOOLS],
    )

    @retry_on_transient_error
    def call_gemini():
        return client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=contents,
            config=generate_config,
        )

    while True:
        response = call_gemini()
        candidate_content = response.candidates[0].content
        contents.append(candidate_content)

        function_calls = [part.function_call for part in candidate_content.parts if part.function_call]
        if not function_calls:
            return response.text

        response_parts = []
        for call in function_calls:
            step = _execute_tool(con, call, client)
            (on_step or _print_step)(step)
            response_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": step["result"]})
            )

        contents.append(types.Content(role="user", parts=response_parts))


def main() -> None:
    if not config.GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY not set in .env - see README for how to get a free one.")

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    con = config.get_connection()
    contents: list = []

    print("Weather agent ready. Ask about Copenhagen, London, or New York. Ctrl-C to quit.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not question:
            continue

        contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
        try:
            answer = run_agent_turn(client, con, contents)
        except genai_errors.ClientError as e:
            if e.code == 429:
                print(f"\nAgent: Hit the free-tier rate/quota limit and retries ran out ({e.message}). Try again shortly, or tomorrow if it's the daily cap.\n")
            else:
                print(f"\nAgent: Request failed ({e.message}). Try rephrasing?\n")
            continue
        except genai_errors.ServerError as e:
            print(f"\nAgent: Gemini's servers are having trouble right now, even after retrying ({e.message}). Try again in a moment.\n")
            continue
        print(f"\nAgent: {answer}\n")


if __name__ == "__main__":
    main()
