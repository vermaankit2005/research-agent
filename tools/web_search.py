from langchain_core.tools import tool
from langchain_tavily import TavilySearch

search_tool = TavilySearch(
    max_results=1,
    topic="general",
    include_images=False,
    search_depth="advanced"
)


@tool
def web_search(query: str) -> str:
    """Search the web for the given query and return the results.
    Args:
        query: The search query string
    """
    return search_tool.invoke({"query": query})
