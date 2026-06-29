# Let's create a research graph here.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END

from models.research_agent_state import ResearchState
from nodes.planning_node import planning_node
from nodes.research_node import research_node
from nodes.summary_node import summary_node
from utils import pick_next_pending_sub_topic


def should_continue(state: ResearchState) -> str:
    """Determine if we should continue to the next sub-topic or end"""
    pending = pick_next_pending_sub_topic(state)
    if not pending:
        return "summary"
    return "research"

def create_research_graph():
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("planning", planning_node)
    graph.add_node("research", research_node)
    graph.add_node("summary", summary_node)

    # Define edges
    graph.set_entry_point("planning")
    graph.add_edge("planning", "research")
    graph.add_conditional_edges("research", should_continue, {"research": "research", "summary": "summary"})
    graph.add_edge("summary", END)

    # The checkpointer saves the state after every step, so the graph can
    # remember progress. It must be passed in here, at compile time.
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
