from copy import deepcopy

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from models.research_agent_state import ResearchState
from models.search_results import SearchResultsOutput, SearchResultItemOutput
from tools.llm import get_llm
from tools.web_search import web_search

tools = [web_search]
MAX_TOOL_LOOPS = 3


def research_node(state: ResearchState) -> ResearchState:
    topic = state["topic"]
    sub_topic = state["current_sub_topic"]

    print(f"========== Research started/resumed for the sub-topic:  {sub_topic} ==========")

    # ---------- Phase 1: gather information using tools ----------
    research_system_prompt = f"""
    You are a meticulous research analyst gathering raw evidence for ONE sub-topic of a larger research topic.

    Your output is NOT the final answer. It is evidence that a later step will combine with the findings of other sub-topics into a single report. So your job is to gather — exhaustively and accurately — not to summarize or conclude.

    ## Scope
    - Investigate ONLY the assigned sub-topic. Use the overall topic as framing, but do NOT drift into other sub-topics.
    - Stay narrow and deep rather than broad and shallow.

    ## Search Budget
    - You have at most {MAX_TOOL_LOOPS} web_search calls. Spend them deliberately.
    - Plan distinct, focused queries that attack the sub-topic from different angles — do NOT repeat near-identical searches.
    - Cross-reference across sources when claims are important or contested.

    ## What to Capture
    - ALL concrete details: facts, statistics, dates, names, definitions, quotes, and figures — do NOT condense or omit detail.
    - Attribute claims to the source they came from.
    - If evidence is thin, conflicting, or unavailable, say so explicitly rather than guessing.

    ## Accuracy
    - Be factual, objective, and neutral. NEVER fabricate facts, figures, or sources.
    - Cite ONLY URLs/sources actually returned by your searches and that you actually used.

    ## Output
    - Finish with a single consolidated findings write-up in plain markdown prose.
    - Prefer completeness over brevity — a later synthesis step handles trimming.
    - If known, mention all the sources used in the findings.
    """

    research_prompt = f"""
    Overall research topic: {topic}
    Sub-topic to research (stay strictly within this): {sub_topic}
    """

    agent = create_agent(get_llm(), tools=tools,
                         middleware=[ModelCallLimitMiddleware(thread_limit=MAX_TOOL_LOOPS + 1, exit_behavior="end")])
    messages = [SystemMessage(research_system_prompt), HumanMessage(research_prompt)]

    final_findings = None
    for chunk in agent.stream({"messages": messages}, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
        final_findings = chunk

    # ---------- Phase 2: map step — condense raw evidence into a clean digest ----------
    print(f"========== Condensing findings for sub-topic ==========")

    gathered = "\n\n".join(
        m.content
        for m in final_findings["messages"]
        if isinstance(m, (AIMessage, ToolMessage)) and m.content
    )

    digest_prompt = """
    You condense raw web-search output into a clean, faithful evidence digest for one sub-topic.

    - Keep EVERY concrete detail: facts, numbers, dates, names, quotes, and the sources used.
    - Drop only noise: tool boilerplate, duplicates, and irrelevant text.
    - Do NOT write report prose, intros, or conclusions — this is evidence, not a report.
    - If the material is thin or empty, state plainly what is missing. Never invent anything.
    """

    digest = get_llm().invoke([
        SystemMessage(digest_prompt),
        HumanMessage(
            f"Overall topic: {topic}\n"
            f"Sub-topic: {sub_topic}\n\n"
            f"Raw material from web searches (may be partial or empty):\n\n"
            f"{gathered if gathered.strip() else '[no usable search results were returned]'}"
        ),
    ])
    raw_findings = digest.content

    response = SearchResultsOutput(research_results=[
        SearchResultItemOutput(search_sub_topic=sub_topic, research_content=raw_findings)
    ])

    print(f"**SubTopic digest**:\n\n{raw_findings}")

    # Lets deep copy the state and make the changes in the copy and pass to the state
    updated_research_sub_topics = deepcopy(state["research_sub_topics"])

    for st in updated_research_sub_topics.sub_topics:
        if st.sub_topic == sub_topic:
            st.status = "done"
            break
    print(f"========== Research completed for {sub_topic}, marking as done ==========")

    # Passing the updated research sub topics to the state, to avoid race condition in parallel execution
    return {"research_results": response,
            "research_sub_topics": updated_research_sub_topics
            }


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
