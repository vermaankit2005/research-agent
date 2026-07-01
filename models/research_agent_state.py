from copy import deepcopy
from typing import Annotated, TypedDict

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


def merge_sub_topics(
        current: SubTopicsOutput | None,
        update: SubTopicsOutput,
) -> SubTopicsOutput:
    """Reducer: OR-merge each branch's 'done' flag so parallel writes don't
    clobber each other. A branch only ever marks its own sub-topic done; we
    trust the update only for what it reports done and keep everything else."""
    if current is None:
        return update
    done = {st.sub_topic for st in update.sub_topics if st.status == "done"}
    merged = deepcopy(current)
    for st in merged.sub_topics:
        if st.sub_topic in done:
            st.status = "done"
    return merged


class ResearchState(TypedDict):
    topic: str
    research_sub_topics: Annotated[SubTopicsOutput, merge_sub_topics]
    research_results: Annotated[SearchResultsOutput, merge_research_results]
    final_report: str
