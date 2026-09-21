"""Shared configuration: MotherDuck connection + Gemini API settings."""

import os
from pathlib import Path

import duckdb
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK_TOKEN")
MOTHERDUCK_DATABASE = os.environ.get("MOTHERDUCK_DATABASE", "weather")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")


def get_connection() -> duckdb.DuckDBPyConnection:
    """
    Connects to the same MotherDuck warehouse the weather-etl-pipeline project
    writes to. No MotherDuck-side read-only enforcement here (that needs a
    paid plan) - safety instead comes entirely from validating every query
    before it runs (see sql_tool.py). Treat this connection as read-write
    at the platform level, and the validator as the thing actually keeping
    it read-only in practice.
    """
    if not MOTHERDUCK_TOKEN:
        raise RuntimeError("MOTHERDUCK_TOKEN not set - add it to .env")

    con = duckdb.connect("md:", config={"motherduck_token": MOTHERDUCK_TOKEN})
    con.execute(f"USE {MOTHERDUCK_DATABASE}")
    con.execute("SET TimeZone = 'UTC'")
    return con
