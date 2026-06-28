from langchain_core.messages import SystemMessage, HumanMessage

from models.research_agent_state import ResearchState
from models.sub_topics_output import SubTopicsOutput
from tools.llm import get_llm_with_structured_output, CONFIG


def planning_node(state: ResearchState) -> ResearchState:
    """ Breaks the given topic into sub-topics, which can be researched independently"""
    original_topic = state["topic"]

    system_prompt = """
    You are a research expert assistant. 
    You deeply understand the given topic and break it down into sub-topics that can be researched independently.
    The sub-topics should be specific and actionable and strictly in ENGLISH.
    
    OUTPUT JSON FORMAT:
    {
        "sub_topics": [
            {
                "sub_topic": str, # The name of the sub-topic
                "search_mode": "websearch", #(always websearch for now)
                "status": "pending" #(always pending when created)
            }
        ]
    }
    """

    user_prompt = f"""
    Given the topic below break it down into sub-topics, which can be researched independently.
    Given topic: {original_topic}
    """

    llm = get_llm_with_structured_output(SubTopicsOutput)
    response = llm.invoke([SystemMessage(system_prompt), HumanMessage(user_prompt)], config=CONFIG)
    print(response)
    for i, sub_topic in enumerate(response.sub_topics):
        print(f"{i + 1}. {sub_topic.sub_topic, sub_topic.search_mode}")

    return {"research_sub_topics": response}


if __name__ == "__main__":
    final_state = planning_node({"topic": "Gen AI"})
    print(final_state)
