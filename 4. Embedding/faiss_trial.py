import pprint
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()
langchain.debug = True

EMBEDDINGS = OpenAIEmbeddings()

# FAISSに保存するテキストデータ
stored_texts = [
    "好きな食べ物は何ですか?",
    "どこにお住まいですか?",
    "朝の電車は混みますね",
    "今日は良いお天気ですね",
    "最近景気悪いですね"]

# OpenAIのEmbeddingsモデルを使ってstored_textsをベクトル化してFAISSに保管
# textファイルではfrom_documentsを使用する。
db = FAISS.from_texts(texts=stored_texts, embedding=EMBEDDINGS)

# Query
query_1 = "今日は雨振らなくてよかった"
found_docs_1 = db.similarity_search(query=query_1, k=4)
# k – Number of Documents to return. Defaults to 4.

print('found_docs_1:')
pprint.pprint(found_docs_1)

print(f"""
==========================================
近傍検索: {query_1}
↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
{found_docs_1[0].page_content}
""")
