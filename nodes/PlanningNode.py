from langchain_core.messages import SystemMessage, HumanMessage

from models.ResearchState import ResearchState
from models.ResearchSubTopics import ResearchSubTopics
from tools.llm import get_llm_with_structured_output, CONFIG


def planning_node(state: ResearchState) -> ResearchState:
    """ Breaks the given topic into sub-topics, which can be researched independently"""
    original_topic = state["topic"]

    system_prompt = """
    You are a research expert assistant. 
    You deeply understand the given topic and break it down into sub-topics that can be researched independently.
    
    OUTPUT JSON FORMAT:
    {
        "sub_topics": [
            {
                "research_sub_topic": str, # The name of the sub-topic
                "search_mode": "websearch", #(always websearch for now)
                "status": "pending" #(always pending when created)
            }
        ]
    }
    """

    prompt = f"""
    Given the topic below break it down into sub-topics, which can be researched independently.
    Given topic: {original_topic}
    """

    llm = get_llm_with_structured_output(ResearchSubTopics)
    response = llm.invoke([SystemMessage(system_prompt), HumanMessage(prompt)], config=CONFIG)
    print(response)
    for i, topic in enumerate(response.sub_topics):
        print(f"{i + 1}. {topic.research_sub_topic, topic.search_mode}")

    return {"research_sub_topics": response}


if __name__ == "__main__":
    final_state = planning_node({"topic": "Gen AI"})
    print(final_state)
