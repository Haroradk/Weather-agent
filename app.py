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

def _answer_rain(df):
    top = df.loc[df["precipitation_sum_mm"].idxmax()]
    top_date = str(top["date"])[:10]
    return (
        f"Over the last {df['date'].nunique()} days, **{top['city']}** had the most rain, "
        f"with **{top['precipitation_sum_mm']:.1f} mm** on {top_date}."
    )


def _answer_wind(df):
    top = df.loc[df["wind_speed_max_kmh"].idxmax()]
    top_date = str(top["date"])[:10]
    return (
        f"The windiest day recently was in **{top['city']}** on {top_date}, "
        f"with a max wind speed of **{top['wind_speed_max_kmh']:.1f} km/h**."
    )


def _answer_forecast(df):
    latest_date = str(df["target_date"].max())[:10]
    rows = df[df["target_date"].astype(str).str.startswith(latest_date)]
    parts = [
        f"**{row.city}** ~{row.predicted_temp_avg_c:.1f}°C, "
        f"{row.predicted_precipitation_sum_mm:.1f} mm rain"
        for row in rows.itertuples()
    ]
    return f"Forecast for {latest_date}: " + "; ".join(parts) + "."


def _answer_temp_range(df):
    parts = [
        f"**{city}**: {g['temp_min_c'].min():.1f}–{g['temp_max_c'].max():.1f}°C"
        for city, g in df.groupby("city")
    ]
    return "Recent range – " + "; ".join(parts) + "."


def _answer_rainy_days(df):
    ranked = df.sort_values("rainy_days", ascending=False)
    top = ranked.iloc[0]
    others = ", ".join(f"{r.city} {r.rainy_days}" for r in ranked.iloc[1:].itertuples())
    return (
        f"**{top['city']}** had the most rainy days: **{top['rainy_days']}** (others: {others}). "
        "The definition of a rainy day came from the warehouse's semantic layer, not this app."
    )


def _rainy_days_sql(con):
    # Built from gold.metric_definitions at run time - change the definition in
    # weather-etl-pipeline's semantic_layer.yml and this query follows it.
    expression, table, filter_sql = con.execute(
        "SELECT expression, table_name, filter FROM gold.metric_definitions WHERE name = 'rainy_days'"
    ).fetchone()
    return f"SELECT city, {expression} AS rainy_days FROM {table} WHERE {filter_sql} GROUP BY city"


def _answer_forecasters_vs_model(df):
    scored = df.dropna(subset=["forecaster_rain_correct", "model_rain_correct"])
    if scored.empty:
        return "No days can be scored yet - check back once more forecast days have passed."
    return (
        f"On the **{len(scored)}** days both can be scored, the NWS forecasters' rain call was right "
        f"**{int(scored['forecaster_rain_correct'].sum())}** times and our ML model's "
        f"**{int(scored['model_rain_correct'].sum())}** times. The forecasters' calls were extracted "
        "from their free-text discussions by an LLM in the pipeline."
    )


DEMO_QUESTIONS = [
    {
        "label": "Who calls rain better: NWS forecasters or our model? (from unstructured text)",
        "sql": (
            "SELECT target_date, forecaster_rain_expected, model_precipitation_sum_mm, "
            "actual_precipitation_sum_mm, forecaster_rain_correct, model_rain_correct "
            "FROM gold.forecaster_vs_model_vs_actual ORDER BY target_date"
        ),
        "answer": _answer_forecasters_vs_model,
    },
    {
        "label": "Which city had the most rainy days? (metric from the semantic layer)",
        "sql": _rainy_days_sql,
        "answer": _answer_rainy_days,
    },
    {
        "label": "Which city had the most rain recently?",
        "sql": (
            "SELECT city, date, precipitation_sum_mm FROM gold.weather_daily_summary "
            "WHERE date <= CURRENT_DATE ORDER BY date DESC LIMIT 9"
        ),
        "answer": _answer_rain,
    },
    {
        "label": "Which city had the strongest wind recently?",
        "sql": (
            "SELECT city, date, wind_speed_max_kmh FROM gold.weather_daily_summary "
            "WHERE date <= CURRENT_DATE ORDER BY date DESC LIMIT 9"
        ),
        "answer": _answer_wind,
    },
    {
        "label": "What's the forecast for tomorrow?",
        "sql": (
            "SELECT city, target_date, predicted_temp_avg_c, predicted_precipitation_sum_mm "
            "FROM gold.weather_forecast WHERE NOT is_backtest ORDER BY target_date DESC LIMIT 3"
        ),
        "answer": _answer_forecast,
    },
    {
        "label": "What was each city's recent temperature range?",
        "sql": (
            "SELECT city, date, temp_min_c, temp_max_c, temp_avg_c FROM gold.weather_daily_summary "
            "WHERE date <= CURRENT_DATE ORDER BY date DESC LIMIT 9"
        ),
        "answer": _answer_temp_range,
    },
]


def run_demo_turn(con, demo_question: dict, on_tool_call) -> str:
    """No LLM call at all - runs one of a fixed set of queries picked from a
    dropdown, and slots the actual returned values into a canned sentence
    template. Lets you see the chat UI mechanics (tool-call expander,
    templated answer) work against real data without spending any Gemini
    quota. The *values* are real; the *sentence* is scripted, not reasoned."""
    sql = demo_question["sql"](con) if callable(demo_question["sql"]) else demo_question["sql"]
    result_text = run_readonly_sql(con, sql)
    on_tool_call(sql, result_text)

    # Re-run to get a DataFrame for the template - run_readonly_sql above
    # returns pre-formatted text for the tool-call bubble, not raw values.
    df = con.execute(sql).df()
    try:
        templated = demo_question["answer"](df)
    except Exception:
        templated = "Got a result but couldn't format a sentence from it - see the raw data above."

    return f"**[Demo mode - no LLM call made]** {templated}"


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
    help="Pick from a fixed list of questions; each runs a real query and fills the values "
    "into a canned sentence, with no LLM call - for exercising the UI without spending Gemini quota.",
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

if demo_mode:
    demo_labels = [q["label"] for q in DEMO_QUESTIONS]
    selected_label = st.selectbox("Pick a question", demo_labels, key="demo_question_select")
    question = selected_label if st.button("Ask") else None
else:
    question = st.chat_input("Ask about the weather data...")

if question:
    with st.chat_message("user"):
        st.write(question)

    con = get_db_connection()
    tool_calls_this_turn = []

    with st.chat_message("assistant"):
        tool_call_area = st.container()

        if demo_mode:
            demo_question = next(q for q in DEMO_QUESTIONS if q["label"] == question)
            # Deliberately isolated from st.session_state.contents (the real
            # Gemini conversation state) - demo turns never touch it, so
            # switching Demo mode off mid-conversation can't leave that
            # history malformed (e.g. two user turns with no model reply
            # between them).
            with st.spinner("Running demo query..."):
                answer = run_demo_turn(
                    con, demo_question,
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
