from typing import TypedDict

from models.FinalReport import FinalReport
from models.ResearchResults import ResearchResults
from models.ResearchSubTopics import ResearchSubTopics


class ResearchState(TypedDict):
    topic: str
    research_sub_topics: ResearchSubTopics
    research_results: ResearchResults
    final_report: FinalReport
