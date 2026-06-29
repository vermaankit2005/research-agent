from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# search_tool = TavilySearch(
#     max_results=1,
#     topic="general",
#     include_images=False,
#     search_depth="advanced"
# )

search_tool = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web for the given query and return the results.
    Args:
        query: The search query string
    """
    return search_tool.invoke({"query": query})
