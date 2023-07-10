import os
import json
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.schema import SystemMessage
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

load_dotenv()
langchain.debug = True

SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
slack_app = App(token=SLACK_BOT_TOKEN)

LLM = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0, verbose=True)

db = FAISS.load_local(folder_path='./',
                      index_name='aws_kendra_index',
                      embeddings=OpenAIEmbeddings())
RETRIEVER = db.as_retriever()


@slack_app.event("app_mention")
def slack_mention_handler(body, say):
    event = body.get('event')
    user_message = event.get('text').split('> ')[-1]
    channel = event.get('channel')
    thread_ts = event.get("thread_ts") or event.get('ts')

    tools = [
        weather_function_tool(),
        google_search_tool(),
        aws_docs_embedding_tool(),
    ]
    # history_memory = create_conversation_memory_with_dynamodb(thread_ts)

    agent = create_openai_functions_agent(prompt=None, tools=tools, memory=None)

    ai_answer = agent.run(user_message)
    say(text=ai_answer, channel=channel, thread_ts=thread_ts)


def google_search_tool() -> Tool:
    search = GoogleSearchAPIWrapper()
    tool = Tool(
        name='Search',
        description='useful for when you need to answer questions about current events. '
                    'You should ask targeted questions',
        func=search.run)
    return tool


def weather_function(location):
    if location == 'Tokyo' or location == '東京':
        weather = '晴れ'
    elif location == 'Osaka' or location == '大阪':
        weather = '曇'
    elif location == 'Nagoya' or location == '名古屋':
        weather = '雪'
    elif location == 'Fukuoka' or location == '福岡':
        weather = '雨'
    else:
        weather = '不明'

    weather_answer = [{'天気': weather}]
    weather_answer_json = json.dumps(weather_answer)
    return weather_answer_json


def aws_docs_embedding_tool() -> Tool:

    tool = Tool(name='AWS_Documents',
                func=RetrievalQA.from_chain_type(llm=LLM, retriever=RETRIEVER),
                description='useful when you want to answer questions about Amazon Kendra')

    return tool


def weather_function_tool() -> Tool:
    tool = Tool(name='Weather',
                func=weather_function,
                description='天気を知りたい場所を入力。例: 東京')
    return tool


def create_openai_functions_agent(prompt, tools, memory):
    _system_message = system_message()
    # agent_kwargs = {
    #     # 'extra_prompt_messages': [MessagesPlaceholder(variable_name='memory')],
    #     'extra_prompt_messages': [MessagesPlaceholder(variable_name='chat_history')],
    # }

    agent = initialize_agent(tools=tools,
                             llm=LLM,
                             agent=AgentType.OPENAI_FUNCTIONS,
                             agent_kwargs=_system_message,
                             # memory=memory,
                             verbose=True)

    return agent


def system_message():
    _system_message = {
        'system_message': SystemMessage(content='You are a specialist of Amazon Kendra.'),
    }
    return _system_message


# def create_conversation_memory_with_dynamodb(session_id):
#     message_history = DynamoDBChatMessageHistory(table_name='LangChainSessionTable',
#                                                  session_id=session_id)
#
#     buffer_memory = ConversationBufferMemory(memory_key='chat_history',
#                                              # input_key='question',
#                                              chat_memory=message_history)
#     # buffer_memory = ConversationBufferMemory(variable_name='memory',
#     #                                          chat_memory=message_history)
#
#     return buffer_memory
#


if __name__ == "__main__":
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
    # @chatgpt-botにメンションがあると、デコレータ:@slack_app.eventが呼び出されます。
