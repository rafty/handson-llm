import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()
langchain.debug = True

LLM = ChatOpenAI(model_name='gpt-3.5-turbo-0613', temperature=0, verbose=True)


def main():
    input_data = 'Who is the current Prime Minister of Japan?'
    prompt_template = create_prompt_template()
    chain = LLMChain(llm=LLM, prompt=prompt_template, verbose=True)
    response = chain.run(question=input_data)
    print(response)


def create_prompt_template():
    template = """
    The following is a friendly conversation between a user and an assistant.
    The AI is talkative and provides lots of specific details from its context.
    If the AI does not know the answer to a question, it truthfully says it does not know.
    Please answer in Japanese.

    user: {question}
    assistant:"""

    prompt_template = PromptTemplate(input_variables=['question'],
                                     template=template)

    return prompt_template


if __name__ == "__main__":
    main()
