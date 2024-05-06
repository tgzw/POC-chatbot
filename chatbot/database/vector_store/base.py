from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_chroma import Chroma

import os


# local new vector store
embedding_function = SentenceTransformerEmbeddings(
    model_name=os.getenv("RAG_EMBEDDING_MODEL")
)
vector_store = Chroma(embedding_function=embedding_function)


class VectorStoreFactory:
    def get_instance():
        return vector_store
