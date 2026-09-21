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

st.set_page_config(page_title="Weather Agent", page_icon="\U0001F916", layout="centered")


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

if not config.GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not set in .env")
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
    st.session_state.contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    client = get_client()
    con = get_db_connection()
    tool_calls_this_turn = []

    with st.chat_message("assistant"):
        tool_call_area = st.container()
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
