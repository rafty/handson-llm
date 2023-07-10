from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def ingest_docs() -> None:
    # loader = ReadTheDocsLoader(path='langchain-docs/langchain.readthedocs.io/en/latest',
    #                            features='html.parser')
    loader = ReadTheDocsLoader(path='../langchain-docs/langchain.readthedocs.io/en/latest',
                               features='html.parser')
    raw_documents = loader.load()
    print(f'loaded {len(raw_documents)}documents')
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=400,
                                                  chunk_overlap=50,
                                                  separators=['\n\n', '\n', ' ', ''])
    documents = text_spliter.split_documents(documents=raw_documents)
    print(f'Splitted into {len(documents)} chunks')

    for doc in documents:
        old_path = doc.metadata['source']
        new_url = old_path.replace('../langchain-docs', 'https:/')
        doc.metadata.update({'source': new_url})

    print(f'Going to insert {len(documents)} to FAISS')
    # Todo: insert documents to FAISS
    # qa_chain = RetrievalQA.from_chain_type(llm=llm,
    #                                            chain_type='stuff',
    #                                            retriever=retriever,
    #                                            verbose=True,
    #                                            chain_type_kwargs=chain_type_kwargs,
    #                                            return_source_documents=True)
    # Todo: return_source_documentsをTrueにするとSource Urlが返る


if __name__ == '__main__':
    ingest_docs()
