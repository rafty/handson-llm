import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo-0613',
    messages=[
        {
            'role': 'system',
            'content': '日本語で返答してください',
        },
        {
            'role': 'user',
            'content': "Who is the current Prime Minister of Japan?",
        },
        {
            "content": "\u73fe\u5728\u306e\u65e5\u672c\u306e\u9996\u76f8\u306f\u83c5\u7fa9\u5049\uff08\u3059\u304c\u30fb\u3088\u3057\u3072\u3067\uff09\u6c0f\u3067\u3059\u3002",
            "role": "assistant"
        }
        ,
        {
            'role': 'user',
            'content': '彼の出身地は？'
        },
    ],
)

print(f'response: {response}')
print(f'message_content: {response.get("choices")[0].get("message").get("content")}')
