import asyncio

from langchain_core.messages import SystemMessage, HumanMessage

from models.research_agent_state import ResearchState
from models.sub_topics_output import SubTopicsOutput
from tools.llm import get_llm_with_structured_output, get_llm


async def planning_node(state: ResearchState) -> ResearchState:
    """ Breaks the given topic into sub-topics, which can be researched independently"""
    original_topic = state["topic"]
    print(f"========== Planning started for the topic {original_topic} ==========")
    system_prompt = """
    You are an expert research strategist specializing in comprehensive topic decomposition. 
    Your role is to analyze complex research topics and systematically break them down into clearly defined, 
    independently researchable sub-topics that enable thorough investigation.

    ## Core Responsibilities
    - Deconstruct the given research topic into its fundamental components
    - Ensure each sub-topic represents a distinct, non-overlapping dimension of research
    - Generate sub-topics that are specific enough to yield actionable findings
    - Maintain conceptual coherence while enabling parallel research streams

    ## Sub-Topic Requirements
    Each sub-topic must be:
    1. **Specific and Focused**: Narrow enough to be thoroughly researched within a single investigation session
    2. **Actionable**: Immediately researchable with clear starting points and expected outcomes
    3. **Independent**: Can be investigated without requiring completion of other sub-topics
    4. **Comprehensive**: Collectively cover all essential aspects of the main topic
    5. **Non-Redundant**: Avoid significant overlap with other sub-topics in the set

    ## Output Specifications
    - Language: Strictly in ENGLISH only
    - Format: Valid JSON object
    - Search Mode: Currently set to "websearch" for all sub-topics
    - Status: Initialize all as "pending"
    - **Maximum sub-topics: 4**

    ## OUTPUT JSON Structure
    {
        "sub_topics": [
            {
                "sub_topic": "Clear, descriptive name of the research sub-topic",
                "search_mode": "websearch",
                "status": "pending"
            }
        ]
    }
    """

    user_prompt = f"""
    Given the topic below break it down into sub-topics, which can be researched independently.
    Given topic: {original_topic}
    """


    llm = get_llm_with_structured_output(SubTopicsOutput)
    response = await llm.ainvoke([SystemMessage(system_prompt), HumanMessage(user_prompt)])

    print(f"The original topic is: {original_topic}, was broken down into {len(response.sub_topics)} sub-topics")
    for i, sub_topic in enumerate(response.sub_topics):
        print(f"{i + 1}. {sub_topic.sub_topic, sub_topic.search_mode}")

    return {"research_sub_topics": response}


if __name__ == "__main__":
    test_config = {"configurable": {"thread_id": "test"}}
    final_state = asyncio.run(planning_node({"topic": "Gen AI"}))
    print(final_state)
