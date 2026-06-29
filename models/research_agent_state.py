from typing import Annotated, TypedDict

from models.final_report import FinalReportWrapper
from models.search_results import SearchResultsOutput
from models.sub_topics_output import SubTopicsOutput


def merge_research_results(
        current: SearchResultsOutput | None,
        update: SearchResultsOutput,
) -> SearchResultsOutput:
    """Reducer: accumulate each sub-topic's findings instead of overwriting."""
    if current is None:
        return update
    return SearchResultsOutput(
        research_results=current.research_results + update.research_results
    )


class ResearchState(TypedDict):
    topic: str
    research_sub_topics: SubTopicsOutput
    research_results: Annotated[SearchResultsOutput, merge_research_results]
    final_report: FinalReportWrapper
