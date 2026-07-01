import asyncio
import uuid

from agents.research_graph import create_research_graph

CONFIG = {"configurable": {"thread_id": uuid.uuid4().hex[:10]}}


def main():
    graph = create_research_graph()
    # The starting state. Only "topic" is filled in; the nodes fill the rest.
    initial_state = {"topic": "Wife Swapping culture"}
    asyncio.run(graph.ainvoke(initial_state, config=CONFIG))


if __name__ == "__main__":
    main()
