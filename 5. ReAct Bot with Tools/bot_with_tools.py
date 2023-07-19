import os
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
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain.prompts import HumanMessagePromptTemplate

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

load_dotenv()
langchain.debug = True

SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
slack_app = App(token=SLACK_BOT_TOKEN)

LLM = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0, verbose=True)

db = FAISS.load_local(folder_path='../4. Embedding',
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
        google_search_tool(),
        aws_docs_embedding_tool(),
    ]
    history_memory = create_conversation_memory_with_dynamodb(thread_ts)
    chat_prompt = create_chat_prompt_template()
    agent = create_agent(prompt=chat_prompt, tools=tools, memory=history_memory)

    contents = format_prompt(chat_prompt=chat_prompt, user_message=user_message)
    ai_answer = agent.run(contents)

    # Slack App Bot say ai_answer.
    say(text=ai_answer, channel=channel, thread_ts=thread_ts)


def google_search_tool() -> Tool:
    search = GoogleSearchAPIWrapper()
    tool = Tool(
        name='search',
        description='Google search tool. '
                    'Useful when you need to answer questions about recent events.',
        func=search.run)
    return tool


def aws_docs_embedding_tool() -> Tool:
    tool = Tool(name='AWS_Documents',
                description='Useful for when you need to answer questions about Amazon Kendra.',
                func=RetrievalQA.from_chain_type(llm=LLM, retriever=RETRIEVER))
    return tool


def create_chat_prompt_template():
    system_template = "Always give an answer in the Japanese Language."
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt_template = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    return chat_prompt_template


def format_prompt(chat_prompt, user_message):
    # https://python.langchain.com/docs/modules/model_io/prompts/prompt_templates/#chat-prompt-template
    # text: from '{text}'
    formatted_prompt = chat_prompt.format_prompt(text=user_message).to_messages()
    contents = " ".join([message_prompt.content for message_prompt in formatted_prompt])
    return contents


def create_conversation_memory_with_dynamodb(session_id):
    # see: https://python.langchain.com/docs/modules/memory/integrations/dynamodb_chat_message_history
    message_history = DynamoDBChatMessageHistory(table_name='LangChainSessionTable',
                                                 session_id=session_id)

    buffer_memory = ConversationBufferMemory(memory_key='chat_history',
                                             chat_memory=message_history,
                                             return_messages=True)

    return buffer_memory


def create_agent(prompt, tools, memory):

    agent = initialize_agent(tools=tools,
                             llm=LLM,
                             agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
                             message=prompt,
                             memory=memory,
                             verbose=True,
                             handle_parsing_errors=True,
                             max_iterations=4)

    return agent


def system_message():
    _system_message = {
        'system_message': SystemMessage(content='You are a specialist of Amazon Kendra.'),
    }
    return _system_message


if __name__ == "__main__":
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
    # @chatgpt-botにメンションがあると、デコレータ:@slack_app.eventが呼び出されます。
