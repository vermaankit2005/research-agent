from typing import TypedDict

from models.final_report import FinalReport
from models.search_results import SearchResultsOutput
from models.sub_topics_output import SubTopicsOutput


class ResearchState(TypedDict):
    topic: str
    research_sub_topics: SubTopicsOutput
    research_results: SearchResultsOutput
    final_report: FinalReport
