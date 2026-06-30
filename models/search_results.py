from pydantic import BaseModel


class SearchResultItemOutput(BaseModel):
    search_sub_topic: str
    research_content: str


class SearchResultsOutput(BaseModel):
    research_results: list[SearchResultItemOutput]
