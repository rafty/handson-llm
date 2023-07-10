import os
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

load_dotenv()
langchain.debug = True

SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
slack_app = App(token=SLACK_BOT_TOKEN)

LLM = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0, verbose=True)


@slack_app.event("app_mention")
def slack_mention_handler(body, say):

    event = body.get('event')
    message = event.get('text').split('> ')[-1]  # '<@U05AZ949XL2> 今日の天気は？'から抽出
    channel = event.get('channel')
    thread_ts = event.get("thread_ts") or event.get('ts')  # thread_tsがなければtsを取得

    prompt_template = create_prompt_template()

    history_memory = create_conversation_memory_with_dynamodb(thread_ts)

    chain = LLMChain(llm=LLM,
                     prompt=prompt_template,
                     verbose=True,
                     memory=history_memory)

    response = chain.run(question=message)

    say(text=response, channel=channel, thread_ts=thread_ts)


def create_prompt_template():
    template = """
    The following is a friendly conversation between a user and an assistant.
    The AI is talkative and provides lots of specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it does not know.
    Please answer in Japanese.

    Current conversation:
    {chat_history}
    user: {question}
    assistant:"""

    prompt_template = PromptTemplate(input_variables=['chat_history', 'question'],
                                     template=template)

    return prompt_template


def create_conversation_memory_with_dynamodb(session_id):
    message_history = DynamoDBChatMessageHistory(table_name='LangChainSessionTable',
                                                 session_id=session_id)

    buffer_memory = ConversationBufferMemory(memory_key='chat_history',
                                             input_key='question',
                                             chat_memory=message_history)

    return buffer_memory


if __name__ == "__main__":
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
    # @chatgpt-botにメンションがあると、デコレータ:@slack_app.eventが呼び出されます。
