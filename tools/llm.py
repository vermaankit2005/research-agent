import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
CONFIG = {"configurable": {"thread_id": "user_ankit"}}


@lru_cache
def get_llm():
    return ChatOpenAI(
    model="deepseek/deepseek-v4-flash",
    temperature=0.4,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"])


def get_llm_with_structured_output(schema):
    return get_llm().with_structured_output(schema, method="json_mode")


def get_llm_with_tools(tools):
    return get_llm().bind_tools(tools)
