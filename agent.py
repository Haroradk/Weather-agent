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
from sql_tool import run_readonly_sql


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

SYSTEM_PROMPT = """\
You are a weather-data assistant. You answer questions using ONLY real data
from the weather warehouse, via the run_readonly_sql tool - never guess or
make up numbers. If a question can't be answered from these tables, say so.

Available tables (DuckDB/MotherDuck SQL):

gold.weather_daily_summary
  city VARCHAR                    -- 'Copenhagen', 'London', or 'New York'
  date DATE                       -- IMPORTANT: this table blends real settled history
                                   -- (date <= CURRENT_DATE) with Open-Meteo's own forecast
                                   -- for the days ahead (date > CURRENT_DATE) IN THE SAME
                                   -- TABLE. MAX(date) is therefore NOT "today" - it's several
                                   -- days in the future. For any question about what actually
                                   -- happened ("last N days", "so far", "recently", totals,
                                   -- averages, "most rain"), always filter date <= CURRENT_DATE.
                                   -- Only include date > CURRENT_DATE when the user explicitly
                                   -- asks about the forecast/future.
  temp_min_c DOUBLE
  temp_max_c DOUBLE
  temp_avg_c DOUBLE
  precipitation_sum_mm DOUBLE
  wind_speed_max_kmh DOUBLE
  updated_at TIMESTAMP

gold.weather_forecast
  city VARCHAR
  target_date DATE
  predicted_temp_min_c DOUBLE
  predicted_temp_max_c DOUBLE
  predicted_temp_avg_c DOUBLE
  predicted_precipitation_sum_mm DOUBLE
  predicted_wind_speed_max_kmh DOUBLE
  model_type VARCHAR
  training_rows INTEGER
  is_backtest BOOLEAN             -- true = retroactive evaluation, false = a real live prediction
  trained_at TIMESTAMP

Only SELECT queries are allowed - the tool will reject anything else. When
you have your answer, respond in plain, concise English and cite the
specific numbers you found (don't just say "it was warmer", say by how much).
"""

SQL_TOOL = types.Tool(
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
        )
    ]
)


def run_agent_turn(client: genai.Client, con, contents: list, on_tool_call=None) -> str:
    """Runs the tool-use loop for one user turn, mutating `contents` in place
    with everything that happened, and returning the final text answer.

    on_tool_call(sql, result), if given, is called instead of printing -
    lets a UI (e.g. app.py) render the same tool-call transparency the CLI
    prints, without duplicating this loop."""
    generate_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[SQL_TOOL],
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
            if call.name == "run_readonly_sql":
                sql = call.args["sql"]
                result = run_readonly_sql(con, sql)
                if on_tool_call:
                    on_tool_call(sql, result)
                else:
                    print(f"  [tool] run_readonly_sql: {sql}")
                    print(f"  [tool result] {result[:300]}{'...' if len(result) > 300 else ''}")
            else:
                result = f"ERROR: unknown tool {call.name}"
                if on_tool_call:
                    on_tool_call(f"<unknown tool: {call.name}>", result)

            response_parts.append(
                types.Part.from_function_response(name=call.name, response={"result": result})
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
