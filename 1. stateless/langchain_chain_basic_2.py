import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from dotenv import load_dotenv

load_dotenv()
langchain.debug = True


LLM = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0, verbose=True)


def main():

    input_data = 'Who is the current Prime Minister of Japan?'

    prompt_template = create_prompt_template()

    session_id = '888888'  # 暫定で指定する。Query毎に値を変更してください。
    history_memory = create_conversation_memory_with_dynamodb(session_id)

    chain = LLMChain(llm=LLM,
                     prompt=prompt_template,
                     verbose=True,
                     memory=history_memory)

    response = chain.run(question=input_data)
    print(response)


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
    main()
