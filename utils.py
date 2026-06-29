from typing import Optional

from models.research_agent_state import ResearchState


# def pick_next_pending_sub_topic(state: ResearchState) -> Optional[str]:
#     """Pick the next pending sub-topic to research"""
#     pending = [st for st in state["research_sub_topics"].sub_topics if st.status == "pending"]
#     if not pending:
#         return None
#     return pending[0].sub_topic

def pick_next_pending_sub_topic(state: ResearchState) -> Optional[str]:
    """Pick the next pending sub-topic to research"""
    for st in state["research_sub_topics"].sub_topics:
        status = st.status
        if status == "pending":
            return st.sub_topic
    return None
