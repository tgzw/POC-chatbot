from langchain_openai import ChatOpenAI

import os


llm = ChatOpenAI(temperature=0, model=os.getenv("OPENAI_MODEL_NAME"))
