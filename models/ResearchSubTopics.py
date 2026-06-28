from typing import Literal
from pydantic import BaseModel

class ResearchSubTopic(BaseModel):
    research_sub_topic: str
    search_mode: Literal["websearch"] = "websearch"
    status: Literal["pending", "done"] = "pending"

class ResearchSubTopics(BaseModel):
    sub_topics: list[ResearchSubTopic]

