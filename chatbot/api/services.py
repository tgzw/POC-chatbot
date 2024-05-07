from chatbot.database.vector_store.base import VectorStoreFactory

import os

# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chatbot.modules.agents.base.module import agent_executor


def upload_pdf(pdf_paths):
    vector_store = VectorStoreFactory.get_instance()

    # Loading documents
    docs = []
    for file in pdf_paths:
        if file[-4:] == ".pdf":
            try:
                doc = PyPDFLoader(file).load()
                docs.append(doc)
            except:
                print(f"Unable to load {file}")

    # split it into chunks

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splits = text_splitter.split_documents(docs[0])

    # create the open-source embedding function
    # embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # load it into Chroma
    vector_store.add_documents(documents=splits)
    print("Pdf processed!")


def send_message(message):
    parsed_message = {"input": message}
    response = agent_executor(parsed_message)
    return response["output"]
