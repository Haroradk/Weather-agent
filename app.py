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

import datetime

import altair as alt
import pandas as pd
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

import config
from agent import run_agent_turn
from catalog_tool import CATALOG_SCHEMA, METRICS_QUERY, TABLES_QUERY
from sql_tool import run_readonly_query

st.set_page_config(page_title="Weather Agent", page_icon="\U0001F916", layout="centered")

# Same palette as the dashboard (weather-etl-pipeline/app.py), so the two apps read as one product.
CITY_COLORS = {"Copenhagen": "#00412D", "London": "#9BCDA0", "New York": "#4B1932"}
SERIES_COLORS = ["#00412D", "#9BCDA0", "#4B1932"]
WHITE_LIGHT = "\u26AA"
LIGHTS = {"success": "\U0001F7E2", "warning": "\U0001F7E1", "failed": "\U0001F534"}
DASHBOARD_URL = "https://haroradk-weatherapp.streamlit.app"

# Clickable starters for an empty live chat. Unlike DEMO_QUESTIONS these go to
# Gemini, so each one costs a few calls of the free-tier quota.
LIVE_STARTERS = [
    "Who calls rain better: NWS forecasters or our ML model?",
    "Which city had the most rainy days?",
    "What have New York forecasters said about wind or storms lately?",
    "How accurate has the temperature forecast been for each city?",
    "Compare daily average temperature in London and Copenhagen over the last two weeks",
    "What's tomorrow's forecast for each city?",
]

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
        "metric": "rainy_days",
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


def run_demo_turn(con, demo_question: dict, on_step) -> str:
    """No LLM call at all - runs one of a fixed set of queries picked from the
    demo buttons, and slots the actual returned values into a canned sentence
    template. Lets you see the chat UI mechanics (steps, result table, chart)
    work against real data without spending any Gemini quota. The *values*
    are real; the *sentence* is scripted, not reasoned."""
    if demo_question.get("metric"):
        on_step({"tool": "catalog", "label": f"Read the {demo_question['metric']} definition from gold.metric_definitions"})
    sql = demo_question["sql"](con) if callable(demo_question["sql"]) else demo_question["sql"]
    result, df = run_readonly_query(con, sql)
    on_step({"tool": "run_readonly_sql", "label": sql, "arg": sql, "result": result, "df": df})

    try:
        templated = demo_question["answer"](df)
    except Exception:
        templated = "Got a result but couldn't format a sentence from it - see the data below."

    return f"**[Demo mode - no LLM call made]** {templated}"


@st.cache_resource
def get_client():
    return genai.Client(api_key=config.GEMINI_API_KEY)


@st.cache_resource
def get_db_connection():
    return config.get_connection()


# ---------------------------------------------------------------- rendering helpers
def _is_failed(step: dict) -> bool:
    return step.get("result", "").startswith(("QUERY REJECTED", "QUERY FAILED"))


def render_step(step: dict) -> None:
    """One line (plus detail) per step inside the "agent's work" panel."""
    tool = step["tool"]
    if tool == "catalog":
        st.markdown(f"\U0001F4DA **{step['label']}**")
    elif tool == "describe_table":
        st.markdown(f"\U0001F50E **Looked up the columns of** `{step['arg']}`")
    elif tool == "search_forecast_discussions":
        n = len(step.get("sources") or [])
        st.markdown(f"\U0001F4F0 **Searched forecaster texts** for \u201c{step['arg']}\u201d \u00b7 {n} passages")
    elif tool == "run_readonly_sql":
        if _is_failed(step):
            st.markdown("\u274C **SQL was not run**")
        else:
            st.markdown(f"\U0001F9EE **Ran SQL** \u00b7 {len(step['df'])} rows")
        st.code(step["arg"], language="sql")
        if _is_failed(step):
            st.caption(step["result"])
    else:
        st.markdown(f"\u2753 {step['label']}")


def steps_summary(steps: list) -> str:
    n_sql = sum(s["tool"] == "run_readonly_sql" for s in steps)
    n_search = sum(s["tool"] == "search_forecast_discussions" for s in steps)
    parts = [f"{n_sql} SQL quer{'y' if n_sql == 1 else 'ies'}"] if n_sql else []
    if n_search:
        parts.append(f"{n_search} text search{'es' if n_search > 1 else ''}")
    return "How I got this: " + (", ".join(parts) or f"{len(steps)} steps")


def _as_dates(df: pd.DataFrame) -> pd.DataFrame:
    """DATE columns arrive as Python date objects (dtype object); make them real datetimes for charting."""
    df = df.copy()
    for col in df.columns:
        values = df[col].dropna()
        if df[col].dtype == object and len(values) and isinstance(values.iloc[0], datetime.date):
            df[col] = pd.to_datetime(df[col])
    return df


def result_chart(df: pd.DataFrame):
    """A chart when the result's shape suggests one: a line over time if there's
    a date column, bars if it's a handful of labelled values. None otherwise."""
    if len(df) < 2:
        return None
    df = _as_dates(df)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    dates = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if not numeric:
        return None
    has_cities = "city" in df.columns and df["city"].nunique() > 1
    city_color = alt.Color("city:N", scale=alt.Scale(domain=list(CITY_COLORS), range=list(CITY_COLORS.values())))

    if dates:
        x = alt.X(f"{dates[0]}:T", title=None)
        if has_cities:
            y = numeric[0]
            return alt.Chart(df).mark_line(point=True).encode(x=x, y=alt.Y(f"{y}:Q", title=y), color=city_color, tooltip=list(df.columns))
        long = df.melt(id_vars=[dates[0]], value_vars=numeric[:3], var_name="measure")
        color = alt.Color("measure:N", title=None, scale=alt.Scale(range=SERIES_COLORS))
        return alt.Chart(long).mark_line(point=True).encode(x=x, y=alt.Y("value:Q", title=None), color=color, tooltip=[dates[0], "measure", "value"])

    labels = [c for c in df.columns if c not in numeric and df[c].dtype == object]
    if labels and len(df) <= 15:
        label, y = labels[0], numeric[0]
        bars = alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{label}:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{y}:Q", title=y),
            tooltip=list(df.columns),
        )
        # City bars are labelled on the axis already, so their colour needs no legend.
        return bars.encode(color=city_color.legend(None)) if label == "city" else bars.encode(color=alt.value(SERIES_COLORS[0]))
    return None


def render_results(steps: list, key: str) -> None:
    """The data behind the answer: every non-empty query result as a chart
    (when it suits one), a sortable table and a CSV download."""
    frames = [s["df"] for s in steps if s["tool"] == "run_readonly_sql" and s.get("df") is not None and not s["df"].empty]
    if not frames:
        return
    st.markdown("**Data behind this answer**")
    containers = st.tabs([f"Query {i + 1}" for i in range(len(frames))]) if len(frames) > 1 else [st.container()]
    for i, (container, df) in enumerate(zip(containers, frames)):
        with container:
            chart = result_chart(df)
            if chart is not None:
                st.altair_chart(chart.properties(height=260).configure_legend(orient="bottom", labelLimit=0), use_container_width=True)
            st.dataframe(df, hide_index=True, width="stretch")
            st.download_button(
                "Download CSV", df.to_csv(index=False), file_name=f"weather_agent_result_{i + 1}.csv",
                mime="text/csv", key=f"download_{key}_{i}", icon=":material/download:",
            )


def _plain(text: str) -> str:
    """Forecaster text is hard-wrapped at ~66 characters and full of symbols
    that Markdown would treat as formatting (or LaTeX, for $)."""
    text = " ".join(text.split())
    for ch in "\\`*_{}[]<>()#+-.!|$~":
        text = text.replace(ch, "\\" + ch)
    return text


def render_sources(steps: list) -> None:
    seen, sources = set(), []
    for step in steps:
        for src in step.get("sources") or []:
            if (src["issued_at"], src["section"]) not in seen:
                seen.add((src["issued_at"], src["section"]))
                sources.append(src)
    if not sources:
        return
    st.markdown("**Forecaster passages the agent retrieved**")
    for src in sources:
        with st.container(border=True):
            st.caption(
                f"NWS New York \u00b7 {src['section']} \u00b7 issued {src['issued_at']:%d %b %Y %H:%M} UTC "
                f"\u00b7 match {src['score']:.2f}"
            )
            content = src["content"] if len(src["content"]) <= 600 else src["content"][:600] + "..."
            st.markdown(_plain(content))


def render_turn(turn: dict, key: str) -> None:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        with st.status(steps_summary(turn["steps"]), state="complete", expanded=False):
            for step in turn["steps"]:
                render_step(step)
        st.write(turn["answer"])
        render_sources(turn["steps"])
        render_results(turn["steps"], key)


# ---------------------------------------------------------------- sidebar
@st.cache_data(ttl=600)
def load_sidebar_info(_con):
    runs = _con.execute(
        "SELECT status, started_at FROM ops.pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchall()
    latest_date = _con.execute("SELECT MAX(date) FROM gold.weather_daily_summary WHERE date <= CURRENT_DATE").fetchone()[0]
    tables = _con.execute(TABLES_QUERY, [CATALOG_SCHEMA, CATALOG_SCHEMA]).fetchall()
    metrics = _con.execute(METRICS_QUERY).fetchall()
    return (runs[0] if runs else None), latest_date, tables, metrics


def clear_chat() -> None:
    st.session_state.contents = []
    st.session_state.display_history = []


con = get_db_connection()
latest_run, latest_date, catalog_tables, catalog_metrics = load_sidebar_info(con)

with st.sidebar:
    st.header("About this agent")
    st.caption(
        "Gemini writes its own SQL against the same MotherDuck warehouse as the dashboard, and can "
        "search what NWS forecasters wrote. Open \u201cHow I got this\u201d under an answer to see every step."
    )
    demo_mode = st.toggle(
        "Demo mode (no Gemini calls)",
        value=False,
        help="Pick from a fixed list of questions; each runs a real query and fills the values "
        "into a canned sentence, with no LLM call - for exercising the UI without spending Gemini quota.",
    )
    st.button("Clear chat", on_click=clear_chat, width="stretch", icon=":material/refresh:")

    st.divider()
    st.subheader("What it knows")
    if latest_run:
        status, started_at = latest_run
        st.markdown(f"{LIGHTS.get(status, WHITE_LIGHT)} Pipeline **{status}** on {started_at:%d %b %H:%M} UTC")
    st.markdown(f"Data up to **{latest_date:%d %b %Y}**")
    with st.expander(f"{len(catalog_tables)} tables it can query"):
        for name, comment in catalog_tables:
            st.markdown(f"**{name}**")
            st.caption(comment or "(no description)")
    with st.expander(f"{len(catalog_metrics)} metric definitions"):
        for name, label, description, *_ in catalog_metrics:
            st.markdown(f"**{label}** (`{name}`)")
            st.caption(description)
    st.caption("Both lists are read live from the warehouse catalog - the same text the agent is given.")

    st.divider()
    st.caption(f"Charts and pipeline health: [weather dashboard]({DASHBOARD_URL})")

# ---------------------------------------------------------------- chat
st.title("\U0001F916 Weather Agent")
st.caption(
    "Ask about Copenhagen, London or New York weather, forecasts, model accuracy, or what "
    "New York forecasters wrote. Answers come with the data and passages behind them."
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

for i, turn in enumerate(st.session_state.display_history):
    render_turn(turn, key=str(i))


def ask(question: str) -> None:
    st.session_state.pending_question = question


# A starter/demo button click lands here on the rerun it triggers; typing goes through chat_input.
typed = None if demo_mode else st.chat_input("Ask about the weather data...")
question = st.session_state.pop("pending_question", None) or typed

if question:
    with st.chat_message("user"):
        st.write(question)

    steps = []
    with st.chat_message("assistant"):
        status = st.status("Working on it...", expanded=True)

        def on_step(step: dict) -> None:
            steps.append(step)
            with status:
                render_step(step)

        if demo_mode:
            demo_question = next(q for q in DEMO_QUESTIONS if q["label"] == question)
            # Deliberately isolated from st.session_state.contents (the real
            # Gemini conversation state) - demo turns never touch it, so
            # switching Demo mode off mid-conversation can't leave that
            # history malformed (e.g. two user turns with no model reply
            # between them).
            answer = run_demo_turn(con, demo_question, on_step)
            status.update(label=steps_summary(steps), state="complete", expanded=False)
        else:
            client = get_client()
            st.session_state.contents.append(types.Content(role="user", parts=[types.Part(text=question)]))
            try:
                answer = run_agent_turn(client, con, st.session_state.contents, on_step=on_step)
                status.update(label=steps_summary(steps), state="complete", expanded=False)
            except (genai_errors.ClientError, genai_errors.ServerError) as e:
                if isinstance(e, genai_errors.ClientError) and e.code == 429:
                    answer = f"Hit the free-tier rate/quota limit, even after retrying ({e.message})."
                elif isinstance(e, genai_errors.ClientError):
                    answer = f"Request failed ({e.message})."
                else:
                    answer = f"Gemini's servers are having trouble right now, even after retrying ({e.message})."
                status.update(label="Stopped - see below", state="error", expanded=False)

        st.write(answer)
        key = str(len(st.session_state.display_history))
        render_sources(steps)
        render_results(steps, key)

    st.session_state.display_history.append({"question": question, "steps": steps, "answer": answer})

# Starter questions: always offered in demo mode (they're its only input);
# in live mode only on an empty chat, since each click spends Gemini quota.
if demo_mode:
    st.markdown("**Pick a demo question** - real data, scripted sentence, no Gemini call")
    for q in DEMO_QUESTIONS:
        st.button(q["label"], on_click=ask, args=(q["label"],), key=f"demo_{q['label']}", width="stretch")
elif not st.session_state.display_history:
    st.markdown("**Not sure what to ask? Try one of these**")
    columns = st.columns(2)
    for i, q in enumerate(LIVE_STARTERS):
        columns[i % 2].button(q, on_click=ask, args=(q,), key=f"starter_{i}", width="stretch")
    st.caption("Each question uses a few calls of the free Gemini quota.")
