from chatbot.database.vector_store.base import VectorStoreFactory

from langchain.tools.retriever import create_retriever_tool

# Retriever tool from vector store
vector_store = VectorStoreFactory.get_instance()
retriever = vector_store.as_retriever()

retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="document_search",
    description="Searches information in documents uploaded by the user",
)
