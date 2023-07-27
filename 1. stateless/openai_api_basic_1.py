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
            'content': '日本の首相は誰ですか？',
        },
    ],
)

print(f'response: {response}')
print(f'message_content: {response.get("choices")[0].get("message").get("content")}')



