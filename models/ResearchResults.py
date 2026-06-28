from typing import Annotated

from langgraph.graph import add_messages
from pydantic import BaseModel


class ResearchResult(BaseModel):
    research_sub_topic: str
    research_content: str
    sources: list[str] | None = None

class ResearchResults(BaseModel):
    research_results: Annotated [list[ResearchResult], add_messages]

