"""
v1_single_call.py — Phase 1: LLM calls ONE tool (calculator) ONCE.

What you will learn
────────────────────────────────────────────────────────────────────────────
1. A tool is a Python function + a name, description, and JSON schema.
2. llm.bind_tools([...]) tells the model which tools exist.
3. The LLM does NOT run the tool — it returns a tool_call object.
   YOUR CODE calls the tool and sends the result back as a ToolMessage.
4. The description is the most important line the model reads when
   deciding whether and how to call a tool.

Message flow
────────────────────────────────────────────────────────────────────────────
  You                   LLM
   │──── HumanMessage ──►│
   │◄─── AIMessage ──────┘  (contains tool_calls=[...])
   │  (your code runs the tool)
   │──── ToolMessage ───►│
   │◄─── AIMessage ──────┘  (final natural-language answer)

References & tutorials
────────────────────────────────────────────────────────────────────────────
LangChain tool calling guide (bind_tools, AIMessage.tool_calls, ToolMessage)
  → https://python.langchain.com/docs/how_to/tool_calling/

LangChain @tool decorator
  → https://python.langchain.com/docs/how_to/custom_tools/

LangChain ChatMistralAI integration
  → https://python.langchain.com/docs/integrations/chat/mistralai/

Mistral tool / function calling
  → https://docs.mistral.ai/capabilities/function_calling/

LangChain message types (HumanMessage, AIMessage, ToolMessage)
  → https://python.langchain.com/docs/concepts/messages/

python-dotenv (loading .env files)
  → https://pypi.org/project/python-dotenv/
────────────────────────────────────────────────────────────────────────────
"""

import os
import json
from dotenv import load_dotenv

# LangChain message types
# Reference: https://python.langchain.com/docs/concepts/messages/
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Mistral chat model via LangChain
# Reference: https://python.langchain.com/docs/integrations/chat/mistralai/
from langchain_mistralai import ChatMistralAI

# Our safe calculator tool (defined in tools.py)
from tools import calculator

# ─── 0. Load environment variables ───────────────────────────────────────────
# python-dotenv reads key=value pairs from .env into os.environ.
# Reference: https://pypi.org/project/python-dotenv/
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError(
        "MISTRAL_API_KEY not found. "
        "Copy .env.example → .env and add your key.\n"
        "Get a key at: https://console.mistral.ai/"
    )

# ─── 1. Create the LLM and bind tools ────────────────────────────────────────
# bind_tools() attaches the tool's JSON schema to every API call.
# The model reads the schema to decide when and how to call the tool.
#
# Key insight: bind_tools does NOT run tools. It only tells the model
# "these tools exist". The model responds with tool_calls; YOU run them.
#
# ChatMistralAI docs:   https://python.langchain.com/docs/integrations/chat/mistralai/
# bind_tools docs:      https://python.langchain.com/docs/how_to/tool_calling/
# Mistral models list:  https://docs.mistral.ai/getting-started/models/models_overview/
llm = ChatMistralAI(
    model="open-mistral-nemo",          # free-tier model with tool calling support
    api_key=MISTRAL_API_KEY,
    temperature=0,                      # deterministic output — good for tools
)

# After bind_tools, every llm.invoke() call sends the tool schema to Mistral.
# Reference: https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html#langchain_core.language_models.chat_models.BaseChatModel.bind_tools
llm_with_tools = llm.bind_tools([calculator])

# ─── 2. Helper: pretty-print a message ───────────────────────────────────────
def print_message(msg, label: str = "") -> None:
    """Display a LangChain message in a readable format."""
    border = "-" * 60
    print(f"\n{border}")
    if label:
        print(f"  {label}")
    print(f"  Type    : {type(msg).__name__}")

    if isinstance(msg, AIMessage):
        print(f"  Content : {msg.content or '(empty — model is calling a tool)'}")
        # tool_calls is a list of dicts: [{name, args, id}, ...]
        # Reference: https://python.langchain.com/docs/concepts/messages/#aimessage
        if msg.tool_calls:
            print(f"  Tool calls:")
            for tc in msg.tool_calls:
                print(f"    - name : {tc['name']}")
                print(f"      args : {tc['args']}")
                print(f"      id   : {tc['id']}")

    elif isinstance(msg, ToolMessage):
        print(f"  Tool id  : {msg.tool_call_id}")
        print(f"  Result   : {msg.content}")

    elif isinstance(msg, HumanMessage):
        print(f"  Content : {msg.content}")

    print(border)


# ─── 3. Run a single tool-call conversation ───────────────────────────────────
def run(user_question: str) -> None:
    """
    Single-turn agent:
        1. Send HumanMessage to LLM
        2. If LLM wants a tool → run it → send ToolMessage → get final answer
        3. If LLM answers directly → print it
    """
    print(f"\n{'=' * 60}")
    print(f"  QUESTION: {user_question}")
    print(f"{'=' * 60}")

    # ── Turn 1: LLM decides whether to call a tool ────────────────────────────
    # HumanMessage wraps the user's text.
    # Reference: https://python.langchain.com/docs/concepts/messages/#humanmessage
    messages = [HumanMessage(content=user_question)]
    ai_msg   = llm_with_tools.invoke(messages)

    print_message(ai_msg, "TURN 1 — LLM response")

    # ── Does the LLM want to call a tool? ─────────────────────────────────────
    # ai_msg.tool_calls is populated ONLY when the model decides to call a tool.
    # It is an empty list [] when the model answers directly.
    # Reference: https://python.langchain.com/docs/concepts/messages/#aimessage
    if not ai_msg.tool_calls:
        print(f"\n[OK] FINAL ANSWER (no tool needed):\n    {ai_msg.content}")
        return

    # ── Run the tool(s) ───────────────────────────────────────────────────────
    # The LLM returned tool_calls but did NOT run anything.
    # We iterate over tool_calls and invoke each tool ourselves.
    # In Phase 1 there is at most one tool call; Phase 2 handles multiple.
    tool_messages = []
    for tool_call in ai_msg.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\n[TOOL] Running tool: {tool_name}({tool_args})")

        # Map tool name → Python function
        # In later phases this will be a dynamic dict of all tools.
        if tool_name == "calculator":
            # .invoke() is the standard way to call a LangChain tool
            # Reference: https://python.langchain.com/docs/how_to/custom_tools/#invoking-the-tool
            result = calculator.invoke(tool_args)
        else:
            result = f"Unknown tool: {tool_name}"

        print(f"[TOOL] Result: {result}")

        # ToolMessage wraps the tool result and links it back to the tool_call
        # via tool_call_id. The LLM needs this id to correlate request → result.
        # Reference: https://python.langchain.com/docs/concepts/messages/#toolmessage
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
        print_message(tool_messages[-1], "TOOL RESULT")

    # ── Turn 2: send tool results back → LLM produces final answer ────────────
    # The full conversation history must be included:
    #   [HumanMessage, AIMessage(with tool_calls), ToolMessage, ...]
    # Reference: https://python.langchain.com/docs/how_to/tool_calling/#passing-tool-results-to-model
    messages = [HumanMessage(content=user_question), ai_msg] + tool_messages
    final_ai  = llm_with_tools.invoke(messages)

    print_message(final_ai, "TURN 2 — Final LLM response")
    print(f"\n[OK] FINAL ANSWER:\n    {final_ai.content}")


# ─── 4. Experiments ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Experiment A: arithmetic → should trigger the calculator tool
    # Learn: when does the LLM decide to call a tool?
    run("What is 1234 * 5678?")

    print("\n")

    # Experiment B: identity question → should answer directly, NO tool call
    # Learn: the LLM is smart enough to skip tools it doesn't need
    run("Who are you? What can you do?")

    print("\n")

    # Experiment C: multi-step arithmetic → one tool call, chained math
    # Learn: the LLM may compose the full expression before calling the tool
    run("If a train travels 1200 km at 80 km/h, how many hours does it take? "
        "Then multiply by 60 to get minutes.")

