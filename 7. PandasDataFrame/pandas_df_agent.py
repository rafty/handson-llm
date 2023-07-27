import langchain
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType
import pandas as pd
from dotenv import load_dotenv


load_dotenv()
langchain.debug = True

LLM = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0, verbose=True)
df = pd.read_csv('./titanic.csv')


def main(text):
    agent = create_pandas_dataframe_agent(llm=LLM,
                                          df=df,
                                          agent_type=AgentType.OPENAI_FUNCTIONS,
                                          verbose=True)
    llm_answer = agent.run(text)
    print(f'--- Final Answer: {llm_answer} ---')


if __name__ == "__main__":
    text_1 = 'how many rows are there?'
    text_2 = 'how many people have more than 3 siblings'
    text_3 = 'whats the square root of the average age?'
    text_4 = 'このデータについて説明して'
    text_5 = '性別ごとの生存者を棒グラフで表示して'
    text_6 = '男女別の生存率を計算し、棒グラフで描いてください。'

    main(text=text_6)
