from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage

from models.ResearchResults import ResearchResults
from models.ResearchState import ResearchState
from tools.llm import get_llm_with_structured_output, CONFIG, get_llm
from tools.web_search import web_search

TOOLS = {"web_search": web_search}
MAX_TOOL_LOOPS = 5


def research_node(state: ResearchState) -> ResearchState:
    """ Researches each pending sub-topic and produces structured research results."""
    topic = state["topic"]
    sub_topics = state["research_sub_topics"].sub_topics

    pending = [st for st in sub_topics if st.status == "pending"]
    sub_topic_text = pending[0].research_sub_topic

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
    Sub-topic to research: {sub_topic_text}
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
                "research_sub_topic": "The sub-topic",
                "research_content": "The research content in markdown format",
                "sources": ["The sources used for the research"]
            }
        ]
    }
    """

    formatter = get_llm_with_structured_output(ResearchResults)
    response = formatter.invoke(
        [SystemMessage(format_system_prompt), HumanMessage(
            f"research_sub_topic: {sub_topic_text} \n Final Findings: {final_findings["messages"][-1].content}")],
        config=CONFIG,
    )

    for i, result in enumerate(response.research_results):
        print(f"{i + 1}. {result.research_sub_topic} — {len(result.sources)} source(s) \n")

    return {"research_results": response}


if __name__ == "__main__":
    from models.ResearchSubTopics import ResearchSubTopic, ResearchSubTopics

    test_state = {
        "topic": "Gen AI",
        "research_sub_topics": ResearchSubTopics(sub_topics=[
            ResearchSubTopic(research_sub_topic="Transformer architecture"),
            ResearchSubTopic(research_sub_topic="Retrieval-Augmented Generation (RAG)"),
        ]),
    }
    final_state = research_node(test_state)
    for i, result in enumerate(final_state["research_results"]):
        print(f"{i + 1}. {result}\n")
