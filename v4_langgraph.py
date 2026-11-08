"""
v4_langgraph.py - Phase 4: LangGraph stateful graph & human-in-the-loop.

What you will learn
------------------------------------------------------------
1. What a LangGraph StateGraph is and why it replaces the hand-written loop
2. How graph STATE flows between nodes (instead of a mutable `messages` list)
3. The two built-in nodes: an LLM node and a ToolNode
4. Conditional edges - how the graph decides "call tool" vs "stop"
5. Human-in-the-loop (HITL) with interrupt_before so a human can
   approve, reject, or edit a tool call before it executes
6. Checkpointing - LangGraph saves the full graph state after every step
   so you can inspect or resume it later

v2 (plain loop) vs v4 (LangGraph) - what changes?
------------------------------------------------------------
  v2: you write the while-loop, the if/else, the message appending
  v4: LangGraph handles all of that; you declare NODES and EDGES

Concept map
------------------------------------------------------------
  Node       = a Python function that transforms state
  Edge       = a directed connection between nodes
  Conditional edge = an edge whose target depends on runtime state
  State      = a typed dict passed through every node
  Checkpoint = a snapshot of state saved after every node runs

Graph topology for this agent:
                  +--------+
        START --> |  llm   |
                  +--------+
                      |
          +-----------+-----------+
          | (conditional edge)    |
          v                       v
     +--------+               [END]
     |  tools |
     +--------+
          |
          v
       (back to llm)

References & tutorials
------------------------------------------------------------
LangGraph concepts (nodes, edges, state):
  -> https://langchain-ai.github.io/langgraph/concepts/

LangGraph quickstart:
  -> https://langchain-ai.github.io/langgraph/tutorials/introduction/

LangGraph human-in-the-loop:
  -> https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/

LangGraph ToolNode:
  -> https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.ToolNode

LangGraph checkpointing (MemorySaver):
  -> https://langchain-ai.github.io/langgraph/reference/checkpoints/

LangGraph interrupt_before:
  -> https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/

tools_condition helper:
  -> https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.tool_node.tools_condition
------------------------------------------------------------
"""

import os
from dotenv import load_dotenv

# LangChain messages
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage

# Mistral chat model
from langchain_mistralai import ChatMistralAI

# ---- LangGraph imports -------------------------------------------------------
# StateGraph  : the graph builder
# START / END : sentinel nodes marking entry and exit points
# Reference   : https://langchain-ai.github.io/langgraph/concepts/
from langgraph.graph import StateGraph, START, END

# MessagesState: a pre-built state schema with a single key "messages"
# that uses add_messages as its reducer (appends instead of overwrites).
# Reference: https://langchain-ai.github.io/langgraph/concepts/low_level/#messagesstate
from langgraph.graph import MessagesState

# ToolNode     : a pre-built node that runs all tool_calls in the last AIMessage
# tools_condition: returns "tools" if the last message has tool_calls, else END
# Reference   : https://langchain-ai.github.io/langgraph/reference/prebuilt/
from langgraph.prebuilt import ToolNode, tools_condition

# MemorySaver  : in-memory checkpointer - saves state after every node
# Reference   : https://langchain-ai.github.io/langgraph/reference/checkpoints/
from langgraph.checkpoint.memory import MemorySaver

# Our 3 shared tools - same as every other phase
from tools import calculator, get_weather, query_expenses

# -----------------------------------------------------------------------------
# 0. Load environment
# -----------------------------------------------------------------------------
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise EnvironmentError(
        "MISTRAL_API_KEY not found. Add it to agent-tools/.env\n"
        "Get a key at: https://console.mistral.ai/"
    )

# -----------------------------------------------------------------------------
# 1. Tool list
# -----------------------------------------------------------------------------
TOOL_LIST = [calculator, get_weather, query_expenses]

# -----------------------------------------------------------------------------
# 2. LLM bound to tools  (same as v2/v3)
# -----------------------------------------------------------------------------
llm = ChatMistralAI(
    model="open-mistral-nemo",
    api_key=MISTRAL_API_KEY,
    temperature=0,
)
llm_with_tools = llm.bind_tools(TOOL_LIST)

# ReAct system prompt carried forward from v3
REACT_SYSTEM_PROMPT = """You are a helpful assistant that solves problems step-by-step.
Before every tool call write a short Thought explaining WHY you are calling it.
Before the final answer write a Thought summarising what you learned.
Always start your message with 'Thought:'"""

# -----------------------------------------------------------------------------
# 3. Node functions
# -----------------------------------------------------------------------------
# A node is simply a function: state -> dict of updates.
# LangGraph calls nodes one at a time, passing the current state.
# Whatever the node returns is MERGED into the state (not replaced).
#
# MessagesState has one key: "messages" (list[BaseMessage]).
# The add_messages reducer APPENDS new messages rather than replacing the list.
# Reference: https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers

def llm_node(state: MessagesState) -> dict:
    """
    Call the LLM with the current message history.

    Prepends the ReAct system prompt if it is not already in the history
    (i.e. on the very first call), then invokes the LLM.

    Returns: {"messages": [ai_msg]}  - LangGraph appends this to state.messages
    """
    messages: list[BaseMessage] = state["messages"]

    # Inject system prompt at position 0 if not already there
    # (MemorySaver preserves history across calls so we check first)
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=REACT_SYSTEM_PROMPT)] + messages

    ai_msg = llm_with_tools.invoke(messages)
    return {"messages": [ai_msg]}


# ToolNode is a pre-built node - no need to write tool dispatch code.
# It reads the tool_calls from the last AIMessage in state.messages,
# runs each tool, and returns a list of ToolMessages.
#
# This replaces the entire for-loop in v2/v3.
# Reference: https://langchain-ai.github.io/langgraph/reference/prebuilt/#toolnode
tool_node = ToolNode(TOOL_LIST)

# -----------------------------------------------------------------------------
# 4. Build the graph
# -----------------------------------------------------------------------------
# StateGraph(MessagesState) creates a graph whose state schema is MessagesState.
# Reference: https://langchain-ai.github.io/langgraph/concepts/low_level/#stategraph

def build_graph(checkpointer=None):
    """
    Build and compile the agent graph.

    Args:
        checkpointer: optional MemorySaver for state persistence.
                      Required for human-in-the-loop (interrupt_before).

    Graph topology:
        START -> llm_node -> (tools_condition) -> tool_node -> llm_node -> ...
                                               -> END
    """
    builder = StateGraph(MessagesState)

    # Add nodes
    # Reference: https://langchain-ai.github.io/langgraph/how-tos/graph-api-basics/
    builder.add_node("llm",   llm_node)
    builder.add_node("tools", tool_node)

    # Entry edge: graph starts at llm_node
    builder.add_edge(START, "llm")

    # Conditional edge: after llm_node, check if we should call tools or stop
    # tools_condition returns:
    #   "tools" - if the last message in state has tool_calls
    #   END     - if the last message has no tool_calls (final answer ready)
    # Reference: https://langchain-ai.github.io/langgraph/reference/prebuilt/#tools_condition
    builder.add_conditional_edges("llm", tools_condition)

    # After tools run, always go back to the LLM
    builder.add_edge("tools", "llm")

    # Compile the graph - this validates edges and locks the topology
    # interrupt_before=["tools"] is passed at compile time for HITL mode
    return builder.compile(checkpointer=checkpointer)


# -----------------------------------------------------------------------------
# 5. Pretty printer
# -----------------------------------------------------------------------------
def _divider(title: str = "") -> None:
    line = "-" * 64
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)


def _print_state(state: dict) -> None:
    """Print every message in the current graph state."""
    for msg in state["messages"]:
        label = type(msg).__name__
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        tool_call_id = getattr(msg, "tool_call_id", None)

        if tool_calls:
            calls_str = ", ".join(
                f"{tc['name']}({tc['args']})" for tc in tool_calls
            )
            print(f"  [{label}] tool_calls: {calls_str}")
        elif tool_call_id:
            print(f"  [{label}] id={tool_call_id} -> {content[:120]}")
        elif content:
            print(f"  [{label}] {content[:200]}")


# -----------------------------------------------------------------------------
# 6a. Experiment A - Basic graph run (no HITL)
# This is v2's agent loop rewritten as a LangGraph graph.
# LangGraph handles the loop, conditionals, and message appending for you.
# -----------------------------------------------------------------------------
def run_basic(user_question: str) -> None:
    """
    Run the agent graph without human-in-the-loop.
    LangGraph drives the full Reason->Act->Observe cycle automatically.
    """
    print(f"\n{'=' * 64}")
    print(f"  [BASIC] {user_question}")
    print(f"{'=' * 64}")

    graph = build_graph()   # no checkpointer needed for simple run

    # stream() yields state snapshots after each node executes.
    # This lets you observe the graph as it runs step by step.
    # Reference: https://langchain-ai.github.io/langgraph/how-tos/stream-values/
    for step_num, (node_name, state) in enumerate(
        graph.stream(
            {"messages": [HumanMessage(content=user_question)]},
            stream_mode="updates",   # yields {node_name: state_update} dicts
        ),
        start=1,
    ):
        _divider(f"STEP {step_num} - Node: {node_name}")
        _print_state(state[node_name])

    print(f"\n{'=' * 64}  DONE  {'=' * 64}\n")


# -----------------------------------------------------------------------------
# 6b. Experiment B - Human-in-the-loop (HITL)
# The graph pauses BEFORE executing the "tools" node so a human can
# inspect the tool call and decide: approve / reject / quit.
#
# Key concepts:
#   interrupt_before=["tools"] -> graph pauses before tools node
#   checkpointer=MemorySaver() -> state is saved so we can resume
#   graph.get_state(config)    -> read the paused state
#   graph.invoke(None, config) -> resume from the saved checkpoint
#
# Reference: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/
# -----------------------------------------------------------------------------
def run_hitl(user_question: str) -> None:
    """
    Run the agent with human approval before every tool call.

    Flow:
      1. Graph runs llm_node   -> decides to call a tool
      2. Graph PAUSES          -> prints the pending tool call
      3. Human types y/n/q     -> approve, skip, or quit
      4. Graph RESUMES         -> tools node runs (or is skipped)
      5. Repeat until final answer
    """
    print(f"\n{'=' * 64}")
    print(f"  [HITL] {user_question}")
    print(f"{'=' * 64}")

    # MemorySaver stores graph state in RAM keyed by thread_id.
    # This is what makes pause-and-resume possible.
    # Reference: https://langchain-ai.github.io/langgraph/reference/checkpoints/
    memory = MemorySaver()

    # Build and compile with interrupt_before=["tools"].
    # This tells LangGraph to pause the graph right before the "tools" node
    # executes, giving the human a chance to approve or reject the call.
    # Reference: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/
    builder = StateGraph(MessagesState)
    builder.add_node("llm",   llm_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", tools_condition)
    builder.add_edge("tools", "llm")
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["tools"],   # <-- HITL: pause before tools
    )

    # thread_id identifies this conversation in the checkpointer.
    # Different thread_ids = independent conversations.
    config = {"configurable": {"thread_id": "hitl-session-1"}}

    # --- First invocation: run until the first interrupt -------------------
    # Reference: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/
    state = graph.invoke(
        {"messages": [HumanMessage(content=user_question)]},
        config=config,
    )

    # Loop: at each pause, ask the user; resume or stop
    while True:
        # Read the current saved state from the checkpointer
        # Reference: https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.get_state
        snapshot = graph.get_state(config)

        # Check if the graph has finished (no more "next" nodes to run)
        if not snapshot.next:
            # Graph reached END - print the final message
            final_msg = snapshot.values["messages"][-1]
            print(f"\n{'=' * 64}")
            print(f"  FINAL ANSWER")
            print(f"{'=' * 64}")
            print(f"\n{final_msg.content}\n")
            break

        # Graph is paused before "tools" - show the pending tool call(s)
        last_ai = snapshot.values["messages"][-1]
        pending_calls = getattr(last_ai, "tool_calls", [])

        print(f"\n  [PAUSED] LLM wants to call {len(pending_calls)} tool(s):")
        for tc in pending_calls:
            print(f"    -> {tc['name']}({tc['args']})")

        if last_ai.content:
            print(f"\n  [THOUGHT] {last_ai.content}")

        # --- Human decision ---
        answer = input("\n  Approve tool call(s)? [y=yes / n=skip / q=quit]: ").strip().lower()

        if answer == "q":
            print("  [HITL] Aborted by user.")
            break

        elif answer == "n":
            # Skip: inject a fake ToolMessage so the LLM knows the tool was rejected
            # then resume. The LLM will try to answer without tool results.
            from langchain_core.messages import ToolMessage, AIMessage
            skip_messages = []
            for tc in pending_calls:
                skip_messages.append(
                    ToolMessage(
                        content="[Tool call rejected by user]",
                        tool_call_id=tc["id"],
                    )
                )
            # Update the graph state manually to inject the skip messages,
            # then resume from the "llm" node instead of "tools".
            # Reference: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/
            graph.update_state(config, {"messages": skip_messages}, as_node="tools")
            state = graph.invoke(None, config=config)

        else:
            # Approve (y or anything else): resume normally - tools node runs
            # Passing None as input means "continue from checkpoint"
            state = graph.invoke(None, config=config)


# -----------------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # ------------------------------------------------------------------
    # Experiment A: Basic graph run - replaces v2's hand-written loop
    # LangGraph manages the entire Reason -> Act -> Observe cycle.
    # Compare the output with v2 to see the same logic, cleaner code.
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  EXPERIMENT A: Basic LangGraph run")
    print("=" * 64)
    run_basic("What is 15% of 2400?")

    # ------------------------------------------------------------------
    # Experiment B: Parallel tools via LangGraph
    # ToolNode handles multiple tool_calls in one go - same as v2's loop.
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  EXPERIMENT B: Parallel tools")
    print("=" * 64)
    run_basic(
        "What is the weather in Mumbai right now? "
        "Also show me the total expenses per city from the database."
    )

    # ------------------------------------------------------------------
    # Experiment C: Human-in-the-loop
    # The graph pauses before EVERY tool call.
    # Try:
    #   y -> approve (tool runs normally)
    #   n -> reject  (LLM gets "[rejected]" message, tries to answer anyway)
    #   q -> quit    (stop the agent entirely)
    # ------------------------------------------------------------------
    print("\n" + "=" * 64)
    print("  EXPERIMENT C: Human-in-the-loop approval")
    print("=" * 64)
    run_hitl(
        "Which city has the highest total expenses in the database? "
        "What is the weather there right now?"
    )
