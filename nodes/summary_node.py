from langchain_core.messages import SystemMessage, HumanMessage

from models.final_report import FinalReportWrapper
from models.research_agent_state import ResearchState
from tools.llm import get_llm_with_structured_output


# TODO: Need to see incase of a large number of sub-topics, how to handle the LLM context limit.
# For now, we are assuming that the number of sub-topics is small enough to fit in the LLM context.
def summary_node(state: ResearchState):
    print("========== Summary node started ==========")

    summary_content = []

    for research_result in state["research_results"].research_results:
        summary_content.append(
            f"## Sub-topic: {research_result.search_sub_topic}\n\n{research_result.research_content}"
        )

    system_prompt = """
    You are an expert research analyst.

    You have been provided with research findings collected from multiple subtopics.
    
    Your task is to produce a comprehensive research report, NOT a brief summary.
    
    Instructions:
    1. Read all research results carefully.
    2. Combine overlapping information into a coherent narrative.
    3. Preserve important facts, statistics, examples, and technical details.
    4. Do not shorten the content unnecessarily.
    5. Explain relationships between different subtopics where applicable.
    6. Remove duplicate information while keeping unique insights.
    7. Write in a professional report style.
    8. The final report should be approximately 1500-2500 words.
    9. Use markdown headings and subheadings.
    10. The content is RAW research material (web search snippets and notes). Extract any
        URLs or named sources that appear within it and use them for the References section.
    11. End with:
       - Key Findings
       - Challenges
       - Future Trends
       - References (extracted from the research content)
    OUTPUT FORMAT:
    {
        "report": {
            "content": "Summary of the research content",
            "sources": ["The sources used for the research"]
        }
    }
    """

    user_prompt = f"""
    Synthesize the following raw research material into the report:

    {"\n\n".join(summary_content)}
    """

    llm = get_llm_with_structured_output(FinalReportWrapper)
    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

    print("========== Summary node completed ==========")
    # Let's print the final output in the formatted way
    print(f"Summary for the topic {state['topic']}: \n{response.report.content}")
    print(f"Sources used: {response.report.sources}")

    return {"final_report": response}


if __name__ == "__main__":
    from models.search_results import SearchResultItemOutput, SearchResultsOutput

    # Test the summary node — mirrors the real state shape produced by research_node
    state = {
        "topic": "Generative AI",
        "research_results": SearchResultsOutput(research_results=[
            SearchResultItemOutput(
                search_sub_topic="What is Generative AI",
                research_content="Generative AI (Gen AI) refers to artificial intelligence models that can create new content such as text, images, audio, video, and code based on patterns learned from large datasets.",
                sources=["https://openai.com/research", "https://ai.google/discover/generative-ai"],
            ),
            SearchResultItemOutput(
                search_sub_topic="Large Language Models",
                research_content="Large Language Models (LLMs) such as GPT, Gemini, and Llama are trained on vast amounts of text data and can perform tasks including summarization, translation, question answering, and code generation.",
                sources=["https://openai.com", "https://ai.meta.com/llama",
                         "https://deepmind.google/technologies/gemini"],
            ),
            SearchResultItemOutput(
                search_sub_topic="Enterprise use cases",
                research_content="Common enterprise use cases for Gen AI include customer support chatbots, document summarization, knowledge management, software development assistance, content creation, and workflow automation.",
                sources=[
                    "https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-economic-potential-of-generative-ai",
                    "https://www.gartner.com/en/topics/generative-ai"],
            ),
            SearchResultItemOutput(
                search_sub_topic="Retrieval-Augmented Generation",
                research_content="Retrieval-Augmented Generation (RAG) combines information retrieval with LLMs, allowing models to generate responses grounded in external knowledge bases and reducing hallucinations.",
                sources=["https://arxiv.org/abs/2005.11401", "https://python.langchain.com/docs/concepts/rag/"],
            ),
            SearchResultItemOutput(
                search_sub_topic="Challenges",
                research_content="Key challenges in Gen AI include hallucinations, bias, data privacy, copyright concerns, model evaluation, and ensuring responsible AI deployment through governance and human oversight.",
                sources=["https://www.nist.gov/itl/ai-risk-management-framework", "https://www.oecd.org/ai/"],
            ),
        ]),
    }
    result = summary_node(state)
