# Let's create a research graph here.
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langgraph.types import Send

from models.research_agent_state import ResearchState
from nodes.planning_node import planning_node
from nodes.research_node import research_node
from nodes.summary_node import summary_node
from utils import pick_next_pending_sub_topic


def fan_out_research_nodes(state: ResearchState):
    """Fan out subtopics to research node parallely """
    return [
        Send("research", {
            "topic": state["topic"],
            "research_sub_topics": state["research_sub_topics"],
            "current_sub_topic": s.sub_topic,
        })
        for s in state["research_sub_topics"].sub_topics
        if s.status == "pending"
    ]


def create_research_graph():
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("planning", planning_node)
    graph.add_node("research", research_node)
    graph.add_node("summary", summary_node)

    # Define edges
    graph.set_entry_point("planning")
    graph.add_conditional_edges("planning", fan_out_research_nodes, ["research"])
    graph.add_edge("research", "summary")
    graph.add_edge("summary", END)

    # The checkpointer saves the state after every step, so the graph can
    # remember progress. It must be passed in here, at compile time.
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)
