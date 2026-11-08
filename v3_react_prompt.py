"""
v3_react_prompt.py — Phase 3: ReAct pattern & visible reasoning.

What you will learn
────────────────────────────────────────────────────────────────────────────
1. The ReAct pattern:  Reason → Act → Observe → Reason → Act → ...
2. How a system prompt shapes the LLM's behaviour at every step
3. How to make the LLM think out loud before calling a tool
   (the "Thought:" prefix in the response)
4. How AIMessage.content can carry reasoning text AND tool_calls together
5. The difference between v2 (silent loop) and v3 (narrated reasoning)

What is ReAct?
────────────────────────────────────────────────────────────────────────────
ReAct = Reason + Act (Yao et al., 2022)
  Paper: https://arxiv.org/abs/2210.03629

The key idea: before every tool call, the LLM must write a short "Thought"
explaining WHY it is calling that tool. After seeing the tool result it
writes another "Thought" before deciding next step.

This makes the agent's decision process transparent and debuggable:

  Human:   What is the weather in Paris? Then convert °C to °F.
  Thought: I need the current temperature in Paris first. I'll call get_weather.
  Act:     get_weather("Paris")
  Observe: Paris: 18°C, Partly cloudy
  Thought: I have 18°C. Formula: F = C × 9/5 + 32 = 18 × 9/5 + 32 = 64.4°F.
           I can answer directly — no tool needed.
  Answer:  It is 18°C (64.4°F) and partly cloudy in Paris right now.

Message flow (same structure as v2, different prompt)
────────────────────────────────────────────────────────────────────────────
  messages = [SystemMessage, HumanMessage]

  LOOP:
    AIMessage ← llm.invoke(messages)      # may have content AND tool_calls
    if no tool_calls → print Thought + Answer → DONE
    for each tool_call → run tool → ToolMessage
    messages.extend([AIMessage, ToolMessage, ...])
    repeat

References & tutorials
────────────────────────────────────────────────────────────────────────────
ReAct paper (Reason + Act):
  → https://arxiv.org/abs/2210.03629

LangChain system prompt / SystemMessage:
  → https://python.langchain.com/docs/concepts/messages/#systemmessage

LangChain tool calling (bind_tools, AIMessage.tool_calls, ToolMessage):
  → https://python.langchain.com/docs/how_to/tool_calling/

LangChain agent loop pattern:
  → https://python.langchain.com/docs/how_to/agent_executor/

Mistral function / tool calling:
  → https://docs.mistral.ai/capabilities/function_calling/
────────────────────────────────────────────────────────────────────────────
"""

import os
from dotenv import load_dotenv

# LangChain message types
# Reference: https://python.langchain.com/docs/concepts/messages/
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage

# Mistral chat model
# Reference: https://python.langchain.com/docs/integrations/chat/mistralai/
from langchain_mistralai import ChatMistralAI

# All 3 shared tools — same as v2, reused unchanged
from tools import calculator, get_weather, query_expenses

# ─── 0. Load environment & create LLM ────────────────────────────────────────
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError(
        "MISTRAL_API_KEY not found. Add it to agent-tools/.env\n"
        "Get a key at: https://console.mistral.ai/"
    )

# ─── 1. Tool registry ─────────────────────────────────────────────────────────
# Same dict-based dispatch as v2 — no change needed here.
TOOLS = {
    "calculator":     calculator,
    "get_weather":    get_weather,
    "query_expenses": query_expenses,
}
TOOL_LIST = list(TOOLS.values())

# ─── 2. Create LLM with tools ─────────────────────────────────────────────────
llm = ChatMistralAI(
    model="open-mistral-nemo",
    api_key=MISTRAL_API_KEY,
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOL_LIST)


# ─── 3. The ReAct system prompt ───────────────────────────────────────────────
# This is the ONLY meaningful change from v2.
#
# A system prompt is a special instruction inserted at position [0] in the
# message list. It shapes every LLM response in the conversation.
#
# Here we instruct the model to follow the ReAct pattern:
#   • Write a "Thought:" before calling any tool (explains WHY)
#   • Write a "Thought:" before giving the final answer (explains HOW it reasoned)
#   • Never skip the Thought step — make reasoning visible
#
# Key insight: the same agent loop code (v2) + a different system prompt = v3.
# This shows how much behaviour you can change with prompt engineering alone.
#
# SystemMessage reference:
#   https://python.langchain.com/docs/concepts/messages/#systemmessage
# Prompt engineering guide:
#   https://www.promptingguide.ai/techniques/react

REACT_SYSTEM_PROMPT = """You are a helpful assistant that solves problems step-by-step
using the ReAct pattern (Reason + Act).

Before every action, write a short "Thought:" that explains your reasoning:
  - WHY you are calling a particular tool
  - WHAT you expect the tool to return
  - HOW the result will help you answer the question

After receiving tool results, write another "Thought:" before deciding the next step.

Format your responses like this:

  When calling a tool:
    Thought: <your reasoning here>
    (then call the tool)

  When giving the final answer:
    Thought: <summarise what you learned from the tools>
    Answer: <your final natural-language answer>

Rules:
  • Always start your message with "Thought:"
  • Be concise — one or two sentences per Thought
  • Use tools only when necessary; answer directly if you already know
  • Never skip the Thought step
"""

# ─── 4. Pretty printer ────────────────────────────────────────────────────────
def _divider(title: str = "") -> None:
    line = "-" * 64
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


# ─── 5. The ReAct agent loop ──────────────────────────────────────────────────
# The loop logic is IDENTICAL to v2. The only structural differences are:
#   1. messages starts with SystemMessage (the ReAct prompt) at index [0]
#   2. We print ai_msg.content — which now contains the LLM's "Thought:"
#
# This is the core lesson of Phase 3:
#   SAME LOOP + DIFFERENT PROMPT = VERY DIFFERENT AGENT BEHAVIOUR
#
# Reference — building a custom agent loop:
#   https://python.langchain.com/docs/how_to/agent_executor/

MAX_STEPS = 10


def run_agent(user_question: str) -> str:
    """
    Run the ReAct agent loop for a given user question.

    Returns the final natural-language answer.
    Prints every Thought and tool call so you can follow the reasoning.
    """
    print(f"\n{'=' * 64}")
    print(f"  QUESTION: {user_question}")
    print(f"{'=' * 64}")

    # ── Conversation history ──────────────────────────────────────────────────
    # Phase 3 difference #1: SystemMessage at position [0].
    # Every subsequent LLM call receives this prompt, so the model always
    # knows it must write "Thought:" before acting.
    #
    # SystemMessage reference:
    #   https://python.langchain.com/docs/concepts/messages/#systemmessage
    messages: list[BaseMessage] = [
        SystemMessage(content=REACT_SYSTEM_PROMPT),   # ← NEW vs v2
        HumanMessage(content=user_question),
    ]

    for step in range(1, MAX_STEPS + 1):
        _divider(f"STEP {step} — Calling LLM")

        # ── LLM call ─────────────────────────────────────────────────────────
        # Same as v2. Because of the system prompt, the model now returns
        # content like "Thought: I need the weather for Mumbai..." before
        # (or instead of) tool calls.
        ai_msg: AIMessage = llm_with_tools.invoke(messages)

        # ── Phase 3 difference #2: print the Thought ─────────────────────────
        # In v2 we printed "(none — calling tools)" when content was empty.
        # In v3 the model always writes a Thought, so content is rarely empty.
        if ai_msg.content:
            print(f"\n  [THOUGHT]\n  {ai_msg.content}")
        print(f"\n  Tool calls: {len(ai_msg.tool_calls)}")

        # ── Case A: no tool calls → LLM produced the final answer ────────────
        # The model's Thought already printed above. We just extract the Answer.
        if not ai_msg.tool_calls:
            # The LLM was asked to prefix with "Thought:" and end with "Answer:"
            # Try to split them for cleaner display; fall back to full content.
            content = ai_msg.content
            if "Answer:" in content:
                thought_part, answer_part = content.split("Answer:", 1)
                print(f"\n{'=' * 64}")
                print(f"  FINAL ANSWER (after {step} step(s))")
                print(f"{'=' * 64}")
                print(f"\n{answer_part.strip()}\n")
                return answer_part.strip()
            else:
                # Model didn't follow the exact format — still a valid answer
                print(f"\n{'=' * 64}")
                print(f"  FINAL ANSWER (after {step} step(s))")
                print(f"{'=' * 64}")
                print(f"\n{content}\n")
                return content

        # ── Case B: LLM wants to call tools ──────────────────────────────────
        # Append the AIMessage (which carries both the Thought and tool_calls)
        messages.append(ai_msg)

        # ── Run each tool and collect ToolMessages ────────────────────────────
        # Identical to v2 — parallel tool calling is still supported.
        # Reference: https://python.langchain.com/docs/how_to/tool_calling_parallel/
        for tool_call in ai_msg.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id   = tool_call["id"]

            print(f"\n  [TOOL] {t_name}({t_args})")

            if t_name in TOOLS:
                try:
                    result = TOOLS[t_name].invoke(t_args)
                except Exception as exc:
                    result = f"Tool error: {exc}"
            else:
                result = f"Unknown tool: '{t_name}'. Available: {list(TOOLS.keys())}"

            print(f"  [RESULT] {str(result)[:200]}")

            # ToolMessage links the result to the tool_call via id.
            # The model reads this as an "Observation" in the ReAct cycle.
            # Reference: https://python.langchain.com/docs/concepts/messages/#toolmessage
            messages.append(
                ToolMessage(content=str(result), tool_call_id=t_id)
            )

    # ── Max steps exceeded ────────────────────────────────────────────────────
    warning = (
        f"[WARNING] Agent stopped after {MAX_STEPS} steps without a final answer. "
        "Consider simplifying the question or checking tool definitions."
    )
    print(warning)
    return warning


# ─── 6. Experiments ──────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Experiment A — Single tool with visible reasoning
    # Compare to v2 Experiment A: same result, but now you see WHY the
    # LLM chose the calculator tool.
    # Expected output:
    #   Thought: The user wants 15% of 2400. I'll use the calculator.
    #   ⚙️  [TOOL] calculator({'expression': '2400 * 0.15'})
    #   📦 [RESULT] 360.0
    #   Thought: The calculator returned 360.0.
    #   Answer: 15% of 2400 is 360.
    # ------------------------------------------------------------------
    run_agent("What is 15% of 2400?")

    # ------------------------------------------------------------------
    # Experiment B — Sequential tools with narrated reasoning
    # The LLM explains why it fetches weather BEFORE it can calculate.
    # Expected: 2 steps
    #   Step 1 Thought: I need Delhi's temperature first.
    #           → get_weather("Delhi")
    #   Step 2 Thought: Got 32°C. Now I'll compute 15% of 32.
    #           → calculator("32 * 0.15")
    #   Final   Thought: Calculator says 4.8.
    #           Answer: Delhi is 32°C; 15% of that is 4.8°C.
    # ------------------------------------------------------------------
    run_agent(
        "What is the current temperature in Delhi? "
        "Then calculate 15% of that temperature."
    )

    # ------------------------------------------------------------------
    # Experiment C — Parallel tools with reasoning visible
    # The Thought should explain that BOTH tasks are independent so it
    # calls them in one step.
    # ------------------------------------------------------------------
    run_agent(
        "What is the weather in Mumbai right now? "
        "Also show me the total expenses per city from the database."
    )

    # ------------------------------------------------------------------
    # Experiment D — Multi-hop with full Thought chain
    # The most interesting experiment: watch the LLM plan two hops.
    #   Thought 1: "I need to find the highest-spending city first."
    #   → query_expenses(...)
    #   Thought 2: "The DB says Mumbai. Now check Mumbai's weather."
    #   → get_weather("Mumbai")
    #   Thought 3: "I have all the info. Ready to answer."
    # ------------------------------------------------------------------
    run_agent(
        "Which city has the highest total expenses in the database? "
        "What is the weather there right now?"
    )
