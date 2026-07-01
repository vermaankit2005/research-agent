import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()

@lru_cache
def get_llm():
    llm = ChatOpenAI(
        model="deepseek/deepseek-v4-flash",
        temperature=0.4,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    return llm.bind(
        extra_body={
            "provider": {
                "order": ["baidu/fp8"],
                "allow_fallbacks": False,
            }
        }
    )

# @lru_cache
# def get_llm():
#     return ChatGroq(model="qwen/qwen3-32b", temperature=0.8, reasoning_format="hidden")


def get_llm_with_structured_output(schema):
    return get_llm().with_structured_output(schema, method="function_calling")
