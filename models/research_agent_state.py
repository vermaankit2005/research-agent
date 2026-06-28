from typing import TypedDict

from models.final_report import FinalReport
from models.search_results import SearchResultsOutput
from models.sub_topics_output import SubTopicsOutput


class ResearchState(TypedDict):
    topic: str
    research_sub_topics: SubTopicsOutput
    research_results: SearchResultsOutput
    final_report: FinalReport

def pick_next_pending_sub_topic(state: ResearchState) -> str:
    """Pick the next pending sub-topic to research"""
    pending = [st for st in state["research_sub_topics"].sub_topics if st.status == "pending"]
    if not pending:
        return None
    return pending[0].sub_topic