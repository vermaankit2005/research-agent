import asyncio

from langchain_core.messages import SystemMessage, HumanMessage
from rich.console import Console
from rich.live import Live

from models.research_agent_state import ResearchState
from tools.llm import get_llm


# TODO: Need to see incase of a large number of sub-topics, how to handle the LLM context limit.
# For now, we are assuming that the number of sub-topics is small enough to fit in the LLM context.
async def summary_node(state: ResearchState):
    print("========== Summary node started ==========")

    summary_content = []

    for research_result in state["research_results"].research_results:
        summary_content.append(
            f"## Sub-topic: {research_result.search_sub_topic}\n\n{research_result.research_content}"
        )

    system_prompt = """
    You are an expert research analyst writing the final report.

    You are given clean, per-sub-topic evidence digests (already condensed from web research).
    Your job is the synthesis step: combine them into ONE comprehensive, coherent report — not a brief summary, 
    and not a list of separate sub-topic blurbs.

    Instructions:
    1. Synthesize across sub-topics into a single narrative; explain how they relate.
    2. Merge overlapping points and remove duplicates, but preserve every unique fact, statistic, and technical detail.
    3. Do not shorten unnecessarily; aim for roughly 1500-2000 words.
    4. Write in a professional report style using markdown headings and subheadings.
    5. End with these sections: Key Findings, Challenges, Future Trends, References.
    6. Populate the `content` field with the full markdown report.
    7. Populate the `sources` field with the URLs/named sources that appear in the digests — use ONLY sources present in the material; never invent any.
    8. Language: Strictly in ENGLISH only
    9. Keep the language simple and easy to understand, avoiding unnecessary jargon.
    """

    user_prompt = f"""
    Synthesize the following per-sub-topic evidence digests into the final report:

    {"\n\n".join(summary_content)}
    """

    llm = get_llm()
    full_text = ""
    chunk_generator = llm.astream([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    with Live(console=Console(), refresh_per_second=100) as live:
        async for chunk in chunk_generator:
            full_text += chunk.content
            live.update(full_text, refresh=True)


if __name__ == "__main__":
    from models.search_results import SearchResultItemOutput, SearchResultsOutput

    # Test the summary node — mirrors the real state shape produced by research_node
    state = {
        "topic": "Generative AI",
        "research_results": SearchResultsOutput(research_results=[
            SearchResultItemOutput(
                search_sub_topic="What is Generative AI",
                research_content="Generative AI (Gen AI) refers to artificial intelligence models that can create new content such as text, images, audio, video, and code based on patterns learned from large datasets.",
            ),
            SearchResultItemOutput(
                search_sub_topic="Large Language Models",
                research_content="Large Language Models (LLMs) such as GPT, Gemini, and Llama are trained on vast amounts of text data and can perform tasks including summarization, translation, question answering, and code generation.",

            ),
            SearchResultItemOutput(
                search_sub_topic="Enterprise use cases",
                research_content="Common enterprise use cases for Gen AI include customer support chatbots, document summarization, knowledge management, software development assistance, content creation, and workflow automation.",

            ),
            SearchResultItemOutput(
                search_sub_topic="Retrieval-Augmented Generation",
                research_content="Retrieval-Augmented Generation (RAG) combines information retrieval with LLMs, allowing models to generate responses grounded in external knowledge bases and reducing hallucinations.",
            ),
            SearchResultItemOutput(
                search_sub_topic="Challenges",
                research_content="Key challenges in Gen AI include hallucinations, bias, data privacy, copyright concerns, model evaluation, and ensuring responsible AI deployment through governance and human oversight.",
            ),
        ]),
    }
    result = asyncio.run(summary_node(state))
