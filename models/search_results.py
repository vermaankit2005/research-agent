from typing import Annotated

from langgraph.graph import add_messages
from pydantic import BaseModel

class SearchResultItemOutput(BaseModel):
    search_sub_topic: str
    research_content: str
    sources: list[str] | None = None

class SearchResultsOutput(BaseModel):
    research_results: Annotated [list[SearchResultItemOutput], add_messages]

