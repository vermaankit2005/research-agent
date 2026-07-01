from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

search_tool = TavilySearch(
    max_results=3,
    topic="general",
    include_images=False,
    search_depth="basic"
)


@tool
async def web_search(query: str) -> str:
    """Search the web for the given query and return the results.
    Args:
        query: The search query string
    """
    return await search_tool.ainvoke({"query": query})
