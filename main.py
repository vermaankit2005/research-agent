from agents.research_graph import create_research_graph

CONFIG = {"configurable": {"thread_id": "user_ankit"}}


def main():
    graph = create_research_graph()
    # The starting state. Only "topic" is filled in; the nodes fill the rest.
    initial_state = {"topic": "Cricket"}
    graph.invoke(initial_state, config=CONFIG)


if __name__ == "__main__":
    main()
