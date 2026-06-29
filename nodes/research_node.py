from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import SystemMessage, HumanMessage

from models.research_agent_state import ResearchState
from models.search_results import SearchResultsOutput
from tools.llm import get_llm_with_structured_output, get_llm
from tools.web_search import web_search
from utils import pick_next_pending_sub_topic

tools = [web_search]
MAX_TOOL_LOOPS = 2


def research_node(state: ResearchState) -> ResearchState:
    topic = state["topic"]
    sub_topic = pick_next_pending_sub_topic(state)

    print(f"========== Research started/resumed for the sub-topic:  {sub_topic} ==========")

    # ---------- Phase 1: gather information using tools ----------
    research_system_prompt = """
    You are a meticulous research analyst. Your task is to thoroughly investigate the given sub-topic and gather detailed, well-sourced findings.

    ## Research Process
    - Use the web_search tool for any factual or up-to-date information
    - Execute multiple focused searches to ensure depth and coverage
    - Cross-reference information when possible for accuracy

    ## Gathering Guidelines
    - Capture ALL relevant facts, statistics, dates, names, and concrete details — do NOT condense or omit detail
    - Use simple, clear language without sacrificing accuracy
    - Maintain strict objectivity and neutrality
    - Be factual and specific — do NOT fabricate facts or sources
    - Structure findings with markdown formatting for readability

    ## Source Requirements
    - List ONLY URLs/sources you actually used
    - Do not cite sources you haven't directly referenced

    ## Constraints
    - Be comprehensive: prefer completeness over brevity. Do NOT summarize away detail — a later step handles synthesis.
    - Plain prose with markdown formatting
    """

    research_prompt = f"""
    Overall research topic: {topic}
    Sub-topic to research: {sub_topic}
    """

    agent = create_agent(get_llm(), tools=tools,
                         middleware=[ModelCallLimitMiddleware(thread_limit=MAX_TOOL_LOOPS + 1, exit_behavior="end")])
    messages = [SystemMessage(research_system_prompt), HumanMessage(research_prompt)]
    final_findings = None
    for chunk in agent.stream({"messages": messages}, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
        final_findings = chunk

    # ---------- Phase 2: format the gathered research into the schema ----------

    print(f"========== Formatting Findings for Subtopic ==========")

    format_system_prompt = """
    You are given a research conversation. Extract the findings and convert them into the required structured format.

    This is a STRUCTURING step, not a summarization step.

    ## Instructions
    - Identify each sub-topic's research findings from the conversation
    - Preserve ALL facts, statistics, dates, names, and details — do NOT condense, shorten, or summarize the content
    - Reorganize the gathered material into clean markdown; reword only for clarity, never to compress
    - Include every cited source
    - Output must be valid JSON only

    ## OUTPUT JSON FORMAT
    {
        "research_results": [
            {
                "search_sub_topic": "The exact sub-topic name",
                "research_content": "The full, detailed findings in markdown format",
                "sources": ["List of all sources referenced"]
            }
        ]
    }
    """

    formatter = get_llm_with_structured_output(SearchResultsOutput)
    conversation = "\n\n".join(
        f"[{m.type}] {m.content}"
        for m in final_findings["messages"]
        if m.content
    )
    response = formatter.invoke([
        SystemMessage(format_system_prompt),
        HumanMessage(f"research_sub_topic: {sub_topic}\n\nResearch conversation:\n{conversation}"),
    ])

    for i, result in enumerate(response.research_results):
        if result.research_content:
            print(f"**SubTopic Content**:\n\n{result.research_content}")
        if result.sources:
            print(f"SubTopic Sources:\n\n {result.sources}")

    ## Marking the topic as done.
    for st in state["research_sub_topics"].sub_topics:
        if st.sub_topic == sub_topic:
            st.status = "done"
            break
    print(f"========== Research completed for {sub_topic}, marking as done ==========")

    return {**state, "research_results": response}


if __name__ == "__main__":
    from models.sub_topics_output import SubTopicItemOutput, SubTopicsOutput

    test_state = {
        "topic": "Gen AI",
        "research_sub_topics": SubTopicsOutput(sub_topics=[
            SubTopicItemOutput(sub_topic="Transformer architecture"),
            SubTopicItemOutput(sub_topic="Retrieval-Augmented Generation (RAG)"),
        ]),
    }
    test_config = {"configurable": {"thread_id": "test"}}
    final_state = research_node(test_state)
    for i, result in enumerate(final_state["research_results"]):
        print(f"{i + 1}. {result}\n")
