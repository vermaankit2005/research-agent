from typing import Literal

from pydantic import BaseModel


class SubTopicItemOutput(BaseModel):
    sub_topic: str
    search_mode: Literal["websearch"] = "websearch"
    status: Literal["pending", "done"] = "pending"


class SubTopicsOutput(BaseModel):
    sub_topics: list[SubTopicItemOutput]
