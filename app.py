"""
Chat frontend for the weather agent - the exact same tool-use loop as
agent.py (imported, not duplicated), wrapped in Streamlit's chat UI.

Run with: streamlit run app.py
"""

import os

import streamlit as st

# Same Streamlit Cloud secrets bridge as weather-etl-pipeline/app.py -
# harmless locally (.env covers that), useful if this ever gets deployed.
try:
    for key in ("MOTHERDUCK_TOKEN", "MOTHERDUCK_DATABASE", "GEMINI_API_KEY", "GEMINI_MODEL"):
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    pass

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import config
from agent import run_agent_turn
from sql_tool import run_readonly_sql

st.set_page_config(page_title="Weather Agent", page_icon="\U0001F916", layout="centered")

DEMO_QUERIES = [
    (
        ("rain", "precip", "wet"),
        "SELECT city, date, precipitation_sum_mm FROM gold.weather_daily_summary "
        "WHERE date <= CURRENT_DATE ORDER BY date DESC LIMIT 9",
    ),
    (
        ("wind",),
        "SELECT city, date, wind_speed_max_kmh FROM gold.weather_daily_summary "
        "WHERE date <= CURRENT_DATE ORDER BY date DESC LIMIT 9",
    ),
    (
        ("forecast", "tomorrow", "predict"),
        "SELECT city, target_date, predicted_temp_avg_c, predicted_precipitation_sum_mm "
        "FROM gold.weather_forecast WHERE NOT is_backtest ORDER BY target_date DESC LIMIT 3",
    ),
]
DEMO_DEFAULT_QUERY = (
    "SELECT city, date, temp_min_c, temp_max_c, temp_avg_c FROM gold.weather_daily_summary "
    "WHERE date <= CURRENT_DATE ORDER BY date DESC LIMIT 9"
)


def run_demo_turn(con, question: str, on_tool_call) -> str:
    """No LLM call at all - picks one hardcoded query by simple keyword
    matching, runs it for real, and returns a canned answer. Lets you see
    the chat UI mechanics (tool-call expander, bubbles) work against real
    data without spending any Gemini quota. Not reasoning - don't mistake
    the "picked query" for the model understanding your question."""
    question_lower = question.lower()
    sql = DEMO_DEFAULT_QUERY
    for keywords, candidate_sql in DEMO_QUERIES:
        if any(keyword in question_lower for keyword in keywords):
            sql = candidate_sql
            break

    result = run_readonly_sql(con, sql)
    on_tool_call(sql, result)
    return (
        "**[Demo mode - no LLM call made]** This is real data from a hardcoded "
        "query chosen by simple keyword matching on your question, not an LLM "
        "reasoning about it. Expand the tool-call above to see the query and result. "
        "Turn off Demo mode in the sidebar to ask the real agent."
    )


@st.cache_resource
def get_client():
    return genai.Client(api_key=config.GEMINI_API_KEY)


@st.cache_resource
def get_db_connection():
    return config.get_connection()


st.title("Weather Agent")
st.caption(
    "Ask about Copenhagen, London, or New York weather. The agent writes its own SQL "
    "against the live warehouse - expand a tool-call bubble to see exactly what it ran."
)

demo_mode = st.sidebar.checkbox(
    "Demo mode (no Gemini calls)",
    value=False,
    help="Runs a hardcoded query picked by keyword-matching your question, against real "
    "data, with no LLM call - for exercising the UI without spending Gemini quota.",
)

if not demo_mode and not config.GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not set in .env (or turn on Demo mode in the sidebar)")
    st.stop()

# Two parallel histories: `contents` is the real conversation state sent to
# Gemini on every call (it needs the raw function_call/function_response
# parts to stay coherent); `display_history` is a simplified structure just
# for rendering the chat UI, since walking the raw Content/Part objects for
# display would be messier than tracking what we need to show separately.
if "contents" not in st.session_state:
    st.session_state.contents = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []


def render_tool_calls(tool_calls: list) -> None:
    for sql, result in tool_calls:
        label = sql if len(sql) <= 80 else sql[:80] + "..."
        with st.expander(f"\U0001F527 {label}"):
            st.code(sql, language="sql")
            st.text(result)


for turn in st.session_state.display_history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        render_tool_calls(turn["tool_calls"])
        st.write(turn["answer"])

question = st.chat_input("Ask about the weather data...")
if question:
    with st.chat_message("user"):
        st.write(question)

    con = get_db_connection()
    tool_calls_this_turn = []

    with st.chat_message("assistant"):
        tool_call_area = st.container()

        if demo_mode:
            # Deliberately isolated from st.session_state.contents (the real
            # Gemini conversation state) - demo turns never touch it, so
            # switching Demo mode off mid-conversation can't leave that
            # history malformed (e.g. two user turns with no model reply
            # between them).
            with st.spinner("Running demo query..."):
                answer = run_demo_turn(
                    con, question,
                    on_tool_call=lambda sql, result: tool_calls_this_turn.append((sql, result)),
                )
        else:
            client = get_client()
            st.session_state.contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
            with st.spinner("Thinking..."):
                try:
                    answer = run_agent_turn(
                        client, con, st.session_state.contents,
                        on_tool_call=lambda sql, result: tool_calls_this_turn.append((sql, result)),
                    )
                except genai_errors.ClientError as e:
                    answer = (
                        f"Hit the free-tier rate/quota limit, even after retrying ({e.message})."
                        if e.code == 429
                        else f"Request failed ({e.message})."
                    )
                except genai_errors.ServerError as e:
                    answer = f"Gemini's servers are having trouble right now, even after retrying ({e.message})."

        with tool_call_area:
            render_tool_calls(tool_calls_this_turn)
        st.write(answer)

    st.session_state.display_history.append(
        {"question": question, "tool_calls": tool_calls_this_turn, "answer": answer}
    )
