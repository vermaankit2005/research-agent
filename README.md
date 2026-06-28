# Research Agent

An AI-powered research agent that takes a topic, breaks it into independently
researchable sub-topics, gathers information from the web for each one, and
compiles the findings into a single structured report.

Built with [LangGraph](https://langchain-ai.github.io/langgraph/) /
[LangChain](https://python.langchain.com/), [Pydantic](https://docs.pydantic.dev/)
for structured outputs, and [Tavily](https://tavily.com/) for web search.

## How it works

The agent is organized as a set of graph nodes, each operating on a shared
`ResearchState`:

| Node | File | Responsibility |
|------|------|----------------|
| **Planning** | `nodes/planning_node.py` | Breaks the topic into independent sub-topics (`SubTopicsOutput`). |
| **Research** | `nodes/research_node.py` | Picks the next pending sub-topic, researches it with the `web_search` tool, and returns structured findings (`SearchResultsOutput`). |
| **Summary** | `nodes/summary_node.py` | Merges all research results into a single comprehensive report (`FinalReport`). |

State flows through the pipeline as:

```
topic → research_sub_topics → research_results → final_report
```

The shared state and helpers live in `models/research_agent_state.py`, and the
structured-output schemas live alongside it in `models/`.

## Project structure

```
ResearchAgent/
├── main.py                  # Entry point
├── models/
│   ├── research_agent_state.py   # ResearchState + helpers
│   ├── sub_topics_output.py      # Planning output schema
│   ├── search_results.py         # Research output schema
│   └── final_report.py           # Final report schema
├── nodes/
│   ├── planning_node.py
│   ├── research_node.py
│   └── summary_node.py
└── tools/
    ├── llm.py               # LLM factory (structured output / tool binding)
    └── web_search.py        # Tavily web-search tool
```

## Setup

Requires **Python 3.13+**.

1. Clone the repo and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or, using [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

2. Create a `.env` file in the project root with the required API keys:

   ```env
   OPENROUTER_API_KEY=your_openrouter_key
   TAVILY_API_KEY=your_tavily_key
   GROQ_API_KEY=your_groq_key
   ```

## Usage

Each node can be run on its own for testing (run as a module from the project
root so package imports resolve):

```bash
python -m nodes.planning_node
python -m nodes.research_node
python -m nodes.summary_node
```

## Configuration

The LLM is configured in `tools/llm.py` (model, temperature, and provider base
URL). It currently routes through OpenRouter via the OpenAI-compatible client;
adjust `get_llm()` to point at a different model or provider.

## Status

Early stage — the individual nodes and models are in place. Wiring them into a
single end-to-end LangGraph pipeline in `main.py` is the next step.
