from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage

from models.search_results import SearchResultsOutput
from models.research_agent_state import ResearchState, pick_next_pending_sub_topic
from tools.llm import get_llm_with_structured_output, CONFIG, get_llm
from tools.web_search import web_search

TOOLS = {"web_search": web_search}
MAX_TOOL_LOOPS = 1


def research_node(state: ResearchState) -> ResearchState:

    topic = state["topic"]
    sub_topic = pick_next_pending_sub_topic(state)

    # ---------- Phase 1: gather information using tools ----------
    research_system_prompt = """
    You are a meticulous research analyst. Research the given sub-topic thoroughly.
    Use the web_search tool whenever you need up-to-date or factual information.
    Call web_search as many times as needed with focused queries, then write up your
    findings. 
    The research should be comprehensive and detailed with proper formatting in simple language, 
    without compromising on accuracy.
    The research should be unbiased and objective.
    
    Be factual and specific; do NOT invent facts or sources. When you are
    done researching, respond with your findings in plain prose, and list every URL /
    source you actually relied on.
    
    MAX 500 words
    """

    research_prompt = f"""
    Overall research topic: {topic}
    Sub-topic to research: {sub_topic}
    """

    agent = create_agent(get_llm(), tools=list(TOOLS.values()))
    config = {"configurable": {"thread_id": "1"}}
    messages = [SystemMessage(research_system_prompt), HumanMessage(research_prompt)]
    final_findings = agent.invoke({"messages": messages}, config)

    # ---------- Phase 2: format the gathered research into the schema ----------
    format_system_prompt = """
    You are given a research conversation. Convert the final findings into the required
    structured format.
    
    OUTPUT JSON FORMAT:
    {
        "research_results": [
            {
                "search_sub_topic": "The sub-topic",
                "research_content": "The research content in markdown format",
                "sources": ["The sources used for the research"]
            }
        ]
    }
    """

    formatter = get_llm_with_structured_output(SearchResultsOutput)
    response = formatter.invoke(
        [SystemMessage(format_system_prompt), HumanMessage(
            f"research_sub_topic: {sub_topic} \n Final Findings: {final_findings["messages"][-1].content}")],
        config=CONFIG,
    )

    for i, result in enumerate(response.research_results):
        print(f"{i + 1}. {result.search_sub_topic} — {len(result.sources)} source(s) \n")

    return {"research_results": response}


if __name__ == "__main__":
    from models.sub_topics_output import SubTopicItemOutput, SubTopicsOutput

    test_state = {
        "topic": "Gen AI",
        "research_sub_topics": SubTopicsOutput(sub_topics=[
            SubTopicItemOutput(sub_topic="Transformer architecture"),
            SubTopicItemOutput(sub_topic="Retrieval-Augmented Generation (RAG)"),
        ]),
    }
    final_state = research_node(test_state, "Gen AI", "Transformer architecture")
    for i, result in enumerate(final_state["research_results"]):
        print(f"{i + 1}. {result}\n")
