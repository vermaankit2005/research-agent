from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

response = client.chat.completions.create(
    model="deepseek/deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello"}],
    extra_body={
        "provider": {
            "order": ["baidu/fp8"],
            "allow_fallbacks": False,
        }
    },
)

print(response.choices[0].message.content)