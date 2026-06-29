from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage

from models.research_agent_state import ResearchState, pick_next_pending_sub_topic
from models.search_results import SearchResultsOutput
from tools.llm import get_llm_with_structured_output, CONFIG, get_llm
from tools.web_search import web_search

TOOLS = {"web_search": web_search}
MAX_TOOL_LOOPS = 1


def research_node(state: ResearchState) -> ResearchState:
    topic = state["topic"]
    sub_topic = pick_next_pending_sub_topic(state)

    # ---------- Phase 1: gather information using tools ----------
    research_system_prompt = """
    You are a meticulous research analyst. Your task is to thoroughly investigate the given sub-topic and produce a comprehensive, well-structured research summary.

    ## Research Process
    - Use the web_search tool for any factual or up-to-date information
    - Execute multiple focused searches to ensure depth and coverage
    - Cross-reference information when possible for accuracy

    ## Writing Guidelines
    - Use simple, clear language without sacrificing accuracy
    - Maintain strict objectivity and neutrality
    - Be factual and specific — do NOT fabricate facts or sources
    - Structure findings with markdown formatting for readability

    ## Source Requirements
    - List ONLY URLs/sources you actually used
    - Do not cite sources you haven't directly referenced

    ## Constraints
    - Maximum 500 words
    - Plain prose with markdown formatting
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
    You are given a research conversation. Extract the final findings and convert them into the required structured format.

    ## Instructions
    - Identify each sub-topic's research findings from the conversation
    - Format the research content in markdown
    - Include all cited sources
    - Output must be valid JSON only

    ## OUTPUT JSON FORMAT
    {
        "research_results": [
            {
                "search_sub_topic": "The exact sub-topic name",
                "research_content": "The summarized findings in markdown format",
                "sources": ["List of all sources referenced"]
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
