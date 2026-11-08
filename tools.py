"""
tools.py — shared tool definitions used by every version of the agent.

Three tools are defined here:
  1. calculator    — safe ast-based expression evaluator
  2. get_weather   — calls Open-Meteo (free, no API key)
  3. query_expenses — runs a SELECT on the local SQLite expenses DB

References & tutorials
─────────────────────────────────────────────────────────────────────────────
LangChain @tool decorator
  → https://python.langchain.com/docs/how_to/custom_tools/

LangChain tool calling overview (bind_tools, ToolMessage, AIMessage)
  → https://python.langchain.com/docs/how_to/tool_calling/

Mistral function / tool calling
  → https://docs.mistral.ai/capabilities/function_calling/

Python ast module (safe expression parsing)
  → https://docs.python.org/3/library/ast.html

Open-Meteo free weather API
  → https://open-meteo.com/en/docs
  → Geocoding API: https://open-meteo.com/en/docs/geocoding-api

SQLite3 standard library
  → https://docs.python.org/3/library/sqlite3.html
─────────────────────────────────────────────────────────────────────────────
"""

import ast          # safe expression parsing — NEVER use eval()
import operator     # arithmetic operations mapped from ast nodes
import sqlite3      # built-in; no install needed
import requests     # pip install requests

from langchain_core.tools import tool   # @tool decorator
# Docs: https://python.langchain.com/api_reference/core/tools/langchain_core.tools.convert.tool.html

# ─── 1. CALCULATOR ───────────────────────────────────────────────────────────
# Why ast instead of eval?
#   eval("__import__('os').system('rm -rf /')") runs arbitrary code.
#   ast.parse + a whitelist of node types is completely safe.
# Reference: https://docs.python.org/3/library/ast.html#ast.literal_eval

# Map ast operator nodes → actual Python functions
_SAFE_OPS: dict = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.USub: operator.neg,   # unary minus: -5
}


def _eval_node(node: ast.AST) -> float:
    """
    Recursively walk an AST node and evaluate it.
    Raises ValueError for any node type not in the whitelist.

    AST node reference: https://docs.python.org/3/library/ast.html#abstract-grammar
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):          # e.g. 42, 3.14
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.BinOp):             # e.g. 3 + 4
        left  = _eval_node(node.left)
        right = _eval_node(node.right)
        op_fn = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported binary operator: {node.op}")
        return op_fn(left, right)

    if isinstance(node, ast.UnaryOp):           # e.g. -5
        operand = _eval_node(node.operand)
        op_fn   = _SAFE_OPS.get(type(node.op))
        if op_fn is None:
            raise ValueError(f"Unsupported unary operator: {node.op}")
        return op_fn(operand)

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a pure arithmetic expression and return the numeric result.

    Use this tool whenever the user asks for a calculation.
    Supports: +  -  *  /  **  %  and parentheses.
    Examples:
        "1234 * 5678"      →  "7006652.0"
        "(100 + 20) * 0.15"  →  "18.0"

    DO NOT pass code, variables, or function calls — only math expressions.

    LangChain @tool docs: https://python.langchain.com/docs/how_to/custom_tools/
    """
    try:
        # ast.parse with mode='eval' returns an Expression node
        # Reference: https://docs.python.org/3/library/ast.html#ast.parse
        tree   = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree)
        return str(result)
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        return f"Error evaluating expression: {exc}"


# ─── 2. WEATHER ──────────────────────────────────────────────────────────────
# Open-Meteo is free and requires NO API key — perfect for learning.
# How it works:
#   Step 1: Geocoding API converts a city name → (latitude, longitude)
#   Step 2: Forecast API returns current weather for those coordinates
#
# Open-Meteo docs:        https://open-meteo.com/en/docs
# Geocoding API docs:     https://open-meteo.com/en/docs/geocoding-api
# WMO weather code list:  https://open-meteo.com/en/docs#weathervariables

# Human-readable labels for WMO weather interpretation codes
# Full list: https://open-meteo.com/en/docs#weathervariables
_WMO_CODES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Icy fog",
    51: "Light drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail",
}


@tool
def get_weather(city: str) -> str:
    """
    Return the current temperature (°C) and weather condition for a city.

    Use this tool when the user asks about the weather in any city.
    Example cities: "Delhi", "London", "New York", "Tokyo".

    Returns a plain-text summary, e.g.:
        "Delhi: 32°C, Clear sky"

    Open-Meteo API docs: https://open-meteo.com/en/docs
    Geocoding API docs:  https://open-meteo.com/en/docs/geocoding-api
    """
    try:
        # ── Step 1: city name → coordinates ──────────────────────────────────
        # Geocoding API reference: https://open-meteo.com/en/docs/geocoding-api
        geo_url    = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
        geo_resp   = requests.get(geo_url, params=geo_params, timeout=10)
        geo_resp.raise_for_status()
        geo_data   = geo_resp.json()

        if not geo_data.get("results"):
            return f"City not found: '{city}'. Try a different spelling."

        result = geo_data["results"][0]
        lat, lon, resolved_name = result["latitude"], result["longitude"], result["name"]

        # ── Step 2: coordinates → current weather ─────────────────────────────
        # Forecast API reference: https://open-meteo.com/en/docs
        wx_url    = "https://api.open-meteo.com/v1/forecast"
        wx_params = {
            "latitude":        lat,
            "longitude":       lon,
            "current_weather": True,          # returns temperature + weathercode
            "temperature_unit": "celsius",
        }
        wx_resp = requests.get(wx_url, params=wx_params, timeout=10)
        wx_resp.raise_for_status()
        wx_data = wx_resp.json()

        current     = wx_data["current_weather"]
        temp        = current["temperature"]          # °C
        wmo_code    = int(current["weathercode"])     # WMO weather interpretation code
        condition   = _WMO_CODES.get(wmo_code, f"Code {wmo_code}")

        return f"{resolved_name}: {temp}°C, {condition}"

    except requests.exceptions.ConnectionError:
        return "Error: no internet connection."
    except requests.exceptions.Timeout:
        return "Error: weather API timed out."
    except Exception as exc:
        return f"Error fetching weather: {exc}"


# ─── 3. EXPENSE DATABASE ─────────────────────────────────────────────────────
# SQLite is built into Python — no install needed.
# Docs: https://docs.python.org/3/library/sqlite3.html
# Tutorial: https://www.sqlitetutorial.net/sqlite-python/
#
# SAFETY RULES applied here:
#   • Allow-list: only SELECT statements are permitted
#   • Read-only connection (no INSERT/UPDATE/DELETE reaches the DB)
#   • Input validation before execution
#
# SQL injection explained: https://owasp.org/www-community/attacks/SQL_Injection
# Parameterised queries:   https://docs.python.org/3/library/sqlite3.html#sqlite3-placeholders

DB_PATH = "expenses.db"   # created by seed_db.py

@tool
def query_expenses(sql: str) -> str:
    """
    Run a read-only SQL SELECT query against the expenses database and return
    the results as plain text.

    The database has ONE table:
        expenses(id, city, category, amount, date)

    Example queries:
        "SELECT city, SUM(amount) FROM expenses GROUP BY city ORDER BY SUM(amount) DESC"
        "SELECT * FROM expenses WHERE category = 'Travel' LIMIT 5"

    Only SELECT statements are allowed. Any other statement will be rejected.

    SQLite docs:    https://docs.python.org/3/library/sqlite3.html
    SQL tutorial:   https://www.w3schools.com/sql/sql_select.asp
    """
    # ── Safety: allow only SELECT statements ─────────────────────────────────
    # Strip leading whitespace/comments and check the first keyword.
    # This guards against DROP TABLE, DELETE, INSERT, etc.
    # Guardrail pattern reference:
    #   https://owasp.org/www-community/attacks/SQL_Injection
    cleaned = sql.strip().lstrip("-").strip()   # remove leading SQL comments
    if not cleaned.upper().startswith("SELECT"):
        return (
            "Rejected: only SELECT queries are allowed. "
            f"Your query started with: '{cleaned[:20]}...'"
        )

    try:
        # ── Open read-only connection ─────────────────────────────────────────
        # uri=True + ?mode=ro enforces read-only at the OS level
        # Reference: https://docs.python.org/3/library/sqlite3.html#sqlite3.connect
        conn   = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute(sql)

        rows    = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]  # column names
        conn.close()

        if not rows:
            return "Query returned no rows."

        # Format as a simple text table
        header = " | ".join(columns)
        sep    = "-" * len(header)
        lines  = [header, sep] + [" | ".join(str(v) for v in row) for row in rows]
        return "\n".join(lines)

    except sqlite3.OperationalError as exc:
        # Return the error as text so the LLM can self-correct its SQL
        return f"SQL error: {exc}"
    except Exception as exc:
        return f"Database error: {exc}"

