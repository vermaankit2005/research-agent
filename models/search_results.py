from pydantic import BaseModel


class SearchResultItemOutput(BaseModel):
    search_sub_topic: str
    research_content: str
    sources: list[str] | None = None


class SearchResultsOutput(BaseModel):
    research_results: list[SearchResultItemOutput]
