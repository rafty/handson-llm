import os
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import DynamoDBChatMessageHistory
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.agents import ZeroShotAgent
from langchain.agents import AgentExecutor
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
    message = event.get('text').split('> ')[-1]
    channel = event.get('channel')
    thread_ts = event.get("thread_ts") or event.get('ts')

    tools = create_google_search_tool()
    prompt_template = create_react_agent_prompt_template(tools=tools)

    history_memory = create_conversation_memory_with_dynamodb(thread_ts)

    react_agent = create_react_agent(prompt=prompt_template, tools=tools, memory=history_memory)
    response = react_agent.run(question=message)

    say(text=response, channel=channel, thread_ts=thread_ts)


def create_conversation_memory_with_dynamodb(session_id):
    message_history = DynamoDBChatMessageHistory(table_name='LangChainSessionTable',
                                                 session_id=session_id)

    buffer_memory = ConversationBufferMemory(memory_key='chat_history',
                                             input_key='question',
                                             chat_memory=message_history)

    return buffer_memory


def create_google_search_tool():
    search = GoogleSearchAPIWrapper()
    tools = [
        Tool(name='Search',
             # description='Search Google for recent results.',
             description='useful for when you need to answer questions about current events. You should ask targeted questions',
             func=search.run),
    ]

    return tools


def create_react_agent_prompt_template(tools):

    prefix = """
    Have a conversation with a human, answering the following questions as best you can.
    Please answer in Japanese.
    You have access to the following tools:"""

    suffix = """
    Begin!"

    {chat_history}
    Question: {question}
    {agent_scratchpad}"""

    prompt_template = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=['chat_history', 'question', 'agent_scratchpad']
    )
    return prompt_template


def create_react_agent(prompt, tools, memory):

    llm_chain = LLMChain(llm=LLM, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent,
                                                     tools=tools,
                                                     verbose=True,
                                                     memory=memory)
    return agent_chain


if __name__ == "__main__":
    SocketModeHandler(slack_app, SLACK_APP_TOKEN).start()
    # @chatgpt-botにメンションがあると、デコレータ:@slack_app.eventが呼び出されます。
