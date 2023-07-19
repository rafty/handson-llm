import os
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

load_dotenv()
langchain.debug = True

# OpenAI
LLM = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0, verbose=True)
embeddings = OpenAIEmbeddings()

# Slack Token
SLACK_APP_TOKEN = os.getenv('SLACK_APP_TOKEN')
SLACK_BOT_TOKEN = os.getenv('SLACK_BOT_TOKEN')
slack_app = App(token=SLACK_BOT_TOKEN)


@slack_app.event("app_mention")
def slack_mention_handler(body, say):
    event = body.get('event')
    message = event.get('text').split('> ')[-1]  # "<@U05AZ949XL2> メッセージ"からmessageを抽出
    channel = event.get('channel')
    thread_ts = event.get("thread_ts") or event.get('ts')  # thread_tsがなければtsを取得

    retriever = load_embedding_db()
    prompt_template = create_retrieval_qa_template()
    conversation_history_memory = create_conversation_memory_with_dynamodb(thread_ts)
    qa = create_retrieval_qa_chain(prompt_template=prompt_template,
                                   vector_db_retriever=retriever,
                                   memory=conversation_history_memory)

    result = qa(message)

    formatted_response = format_response(result=result)

    say(text=formatted_response, channel=channel, thread_ts=thread_ts)


def create_source_string(result):
    if not result['source_documents']:
        return ''

    source_urls_string = 'sources:\n'
    for i, source_document in enumerate(result['source_documents']):
        source_urls_string += f'{i + 1}. {source_document.metadata["source"]}\n'

    return source_urls_string


def format_response(result):
    source_urls_string = create_source_string(result=result)
    return f"{result['result']}\n\n {source_urls_string}"


def load_embedding_db():
    db = FAISS.load_local(folder_path='./', index_name='aws_kendra_index', embeddings=embeddings)
    retriever = db.as_retriever()
    return retriever


def create_retrieval_qa_template():

    prompt_template = """
    Use the following pieces of context (delimited by <ctx></ctx>) and 
    the chat history (delimited by <hs></hs>) to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Please answer in Japanese.

    <ctx>
    {context}
    </ctx>

    <hs>
    {chat_history}
    </hs>

    Question: {question}
    Helpful Answer:"""

    prompt_template = PromptTemplate(template=prompt_template,
                                     input_variables=['context', 'chat_history', 'question'])
    return prompt_template


def create_conversation_memory_with_dynamodb(session_id):
    message_history = DynamoDBChatMessageHistory(table_name='LangChainSessionTable',
                                                 session_id=session_id)

    buffer_memory = ConversationBufferMemory(memory_key='chat_history',
                                             input_key='question',
                                             chat_memory=message_history)

    return buffer_memory


def create_retrieval_qa_chain(prompt_template, vector_db_retriever, memory):
    chain_type_kwargs = {
        'verbose': True,
        'prompt': prompt_template,
        'memory': memory,
    }

    qa_chain = RetrievalQA.from_chain_type(llm=LLM,
                                           chain_type='stuff',
                                           retriever=vector_db_retriever,
                                           chain_type_kwargs=chain_type_kwargs,
                                           return_source_documents=True,
                                           verbose=True)

    return qa_chain


if __name__ == '__main__':
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
