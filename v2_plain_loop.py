"""
v2_plain_loop.py — Phase 2: The agent loop in plain code.

What you will learn
────────────────────────────────────────────────────────────────────────────
1. The agent loop pattern:
       call LLM → tool calls? → run tools → append results → repeat → answer
2. All 3 tools together: calculator, get_weather, query_expenses
3. Parallel tool calling — multiple tools invoked in a single LLM turn
4. max_steps guard so the agent can never loop forever
5. Tool errors returned as text so the LLM can self-correct on the next turn

Message flow (multi-turn)
────────────────────────────────────────────────────────────────────────────
  messages = [HumanMessage]

  LOOP:
    AIMessage ← llm.invoke(messages)
    if no tool_calls → DONE, print final answer
    for each tool_call → run tool → ToolMessage
    messages.extend([AIMessage, ToolMessage, ...])
    repeat

References & tutorials
────────────────────────────────────────────────────────────────────────────
LangChain agent loop pattern
  → https://python.langchain.com/docs/how_to/agent_executor/

LangChain tool calling (bind_tools, AIMessage.tool_calls, ToolMessage)
  → https://python.langchain.com/docs/how_to/tool_calling/

Parallel tool calling (multiple tools in one LLM turn)
  → https://python.langchain.com/docs/how_to/tool_calling_parallel/

LangChain message types
  → https://python.langchain.com/docs/concepts/messages/

Open-Meteo API (get_weather)
  → https://open-meteo.com/en/docs

SQLite guardrail pattern (SELECT-only allow-list)
  → https://owasp.org/www-community/attacks/SQL_Injection

Mistral function / tool calling
  → https://docs.mistral.ai/capabilities/function_calling/
────────────────────────────────────────────────────────────────────────────
"""

import os
from dotenv import load_dotenv

# LangChain message types
# Reference: https://python.langchain.com/docs/concepts/messages/
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

# Mistral chat model
# Reference: https://python.langchain.com/docs/integrations/chat/mistralai/
from langchain_mistralai import ChatMistralAI

# All 3 shared tools — defined once in tools.py and reused across every phase
from tools import calculator, get_weather, query_expenses

# ─── 0. Load environment & create LLM ────────────────────────────────────────
# python-dotenv docs: https://pypi.org/project/python-dotenv/
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError(
        "MISTRAL_API_KEY not found. Add it to agent-tools/.env\n"
        "Get a key at: https://console.mistral.ai/"
    )

# ─── 1. Tool registry ─────────────────────────────────────────────────────────
# A dict mapping tool name → callable lets us dispatch tool_calls dynamically.
# This is better than hard-coded if/elif — just add a new tool to the dict.
#
# LangChain @tool docs: https://python.langchain.com/docs/how_to/custom_tools/
TOOLS = {
    "calculator":    calculator,
    "get_weather":   get_weather,
    "query_expenses": query_expenses,
}
TOOL_LIST = list(TOOLS.values())   # passed to bind_tools

# ─── 2. Create LLM bound to all 3 tools ──────────────────────────────────────
# bind_tools sends every tool's JSON schema on every API call so the LLM
# knows what's available. It does NOT run the tools.
#
# bind_tools reference: https://python.langchain.com/docs/how_to/tool_calling/
llm = ChatMistralAI(
    model="open-mistral-nemo",
    api_key=MISTRAL_API_KEY,
    temperature=0,                  # deterministic output for tools
)
llm_with_tools = llm.bind_tools(TOOL_LIST)


# ─── 3. Pretty printer ───────────────────────────────────────────────────────
def _divider(title: str = "") -> None:
    line = "-" * 64
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ─── 4. The agent loop ───────────────────────────────────────────────────────
# This is the heart of Phase 2 — a plain Python while-loop that replicates
# what frameworks like LangGraph and AgentExecutor do internally.
#
# Key insight: the loop is just message list management.
#   - Start with [HumanMessage]
#   - Each iteration appends: AIMessage (from LLM) + ToolMessages (from tools)
#   - Stop when the AIMessage has no tool_calls (final answer) or max_steps hit
#
# Reference — building a custom agent loop:
#   https://python.langchain.com/docs/how_to/agent_executor/

MAX_STEPS = 10   # guard: stop after this many LLM calls to prevent infinite loops
                 # Learn more about why this matters:
                 # https://python.langchain.com/docs/concepts/agents/#agent-execution


def run_agent(user_question: str) -> str:
    """
    Run the agent loop for a given user question.

    Returns the final natural-language answer from the LLM.
    Prints a trace of every step so you can see the agent thinking.
    """
    print(f"\n{'=' * 64}")
    print(f"  QUESTION: {user_question}")
    print(f"{'=' * 64}")

    # ── Conversation history starts with just the user's message ─────────────
    # The full history is passed on every LLM call so the model has context.
    # Reference: https://python.langchain.com/docs/concepts/messages/#humanmessage
    messages: list[BaseMessage] = [HumanMessage(content=user_question)]

    for step in range(1, MAX_STEPS + 1):
        _divider(f"STEP {step} — Calling LLM")

        # ── LLM call ─────────────────────────────────────────────────────────
        # The LLM receives the entire conversation history every turn.
        # It returns either:
        #   (a) an AIMessage with content  → final answer, exit loop
        #   (b) an AIMessage with tool_calls → we must run the tools
        #
        # Reference: https://python.langchain.com/docs/how_to/tool_calling/
        ai_msg: AIMessage = llm_with_tools.invoke(messages)

        print(f"  LLM content  : {ai_msg.content or '(none — calling tools)'}")
        print(f"  Tool calls   : {len(ai_msg.tool_calls)}")

        # ── Case A: no tool calls → LLM produced the final answer ────────────
        if not ai_msg.tool_calls:
            print(f"\n{'=' * 64}")
            print(f"  FINAL ANSWER (after {step} step(s))")
            print(f"{'=' * 64}")
            print(f"\n{ai_msg.content}\n")
            return ai_msg.content

        # ── Case B: LLM wants to call tools ──────────────────────────────────
        # Append the AIMessage to history FIRST so the model can see its own
        # tool_call request when we send ToolMessages back.
        messages.append(ai_msg)

        # ── Run each tool and collect ToolMessages ────────────────────────────
        # Parallel tool calling: the LLM can request MULTIPLE tools in one turn.
        # We run them all and append every ToolMessage before the next LLM call.
        #
        # Parallel tool calling reference:
        #   https://python.langchain.com/docs/how_to/tool_calling_parallel/
        for tool_call in ai_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id   = tool_call["id"]

            print(f"\n  [TOOL] {t_name}({t_args})")

            # ── Dispatch: look up the tool function and call it ───────────────
            # Using a dict avoids a long if/elif chain and scales to N tools.
            if t_name in TOOLS:
                try:
                    # .invoke() is the LangChain standard for calling a tool
                    # Reference: https://python.langchain.com/docs/how_to/custom_tools/#invoking-the-tool
                    result = TOOLS[t_name].invoke(t_args)
                except Exception as exc:
                    # Return the error AS TEXT so the LLM can self-correct
                    # on the next iteration instead of crashing the program.
                    # This is a key reliability pattern in agent design.
                    result = f"Tool error: {exc}"
            else:
                # Unknown tool — LLM hallucinated a tool name
                result = f"Unknown tool: '{t_name}'. Available: {list(TOOLS.keys())}"

            print(f"  [RESULT] {str(result)[:200]}")   # truncate long results for readability

            # ToolMessage links the result back to the specific tool_call via id
            # Reference: https://python.langchain.com/docs/concepts/messages/#toolmessage
            messages.append(
                ToolMessage(content=str(result), tool_call_id=t_id)
            )

    # ── Max steps exceeded ────────────────────────────────────────────────────
    # This is a safety net — a well-formed question should finish in 2-4 steps.
    # If we hit this, something has gone wrong (hallucinating tools, bad input).
    #
    # Reference — why max_steps matters:
    #   https://python.langchain.com/docs/concepts/agents/#agent-execution
    warning = (
        f"[WARNING] Agent stopped after {MAX_STEPS} steps without a final answer. "
        "Consider simplifying the question or checking tool definitions."
    )
    print(warning)
    return warning


# ─── 5. Experiments ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Experiment A — Single tool (calculator)
    # Expected: 1 step, 1 tool call
    # Learn   : basic tool use, same as Phase 1
    # Docs    : https://python.langchain.com/docs/how_to/tool_calling/
    # ------------------------------------------------------------------
    run_agent("What is 15% of 2400?")

    # ------------------------------------------------------------------
    # Experiment B — Sequential tools (weather → calculator)
    # Expected: 2 steps
    #   Step 1: get_weather("Delhi") → temperature e.g. 32°C
    #   Step 2: calculator("32 * 0.15") → 4.8
    # Learn   : chained tool use, the LLM decides the order
    # Docs    : https://python.langchain.com/docs/how_to/agent_executor/
    # ------------------------------------------------------------------
    run_agent(
        "What is the current temperature in Delhi? "
        "Then calculate 15% of that temperature."
    )

    # ------------------------------------------------------------------
    # Experiment C — Parallel tools (weather + expenses in one step)
    # Expected: 1-2 steps
    #   Step 1: get_weather("Mumbai") AND query_expenses(sql) in PARALLEL
    #   Step 2: LLM synthesises both results into final answer
    # Learn   : multiple tool_calls in a single AIMessage turn
    # Docs    : https://python.langchain.com/docs/how_to/tool_calling_parallel/
    # ------------------------------------------------------------------
    run_agent(
        "What is the weather in Mumbai right now? "
        "Also show me the total expenses per city from the database."
    )

    # ------------------------------------------------------------------
    # Experiment D — DB + calculator (the Phase project "definition of done")
    # Expected: 2-3 steps
    #   Step 1: query_expenses → find highest spending city (Mumbai)
    #   Step 2: get_weather("Mumbai") → temperature
    #   (Optional Step 3: calculator for any derived figure)
    # Learn   : multi-hop reasoning driven entirely by the LLM
    # ------------------------------------------------------------------
    run_agent(
        "Which city has the highest total expenses in the database? "
        "What is the weather there right now?"
    )

