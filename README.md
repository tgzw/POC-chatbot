

## Deployment
- Put an openai api key in .env
- Change the rest of variables in .env at your will

Use python 3.11.0
```
pip install -r requirements
```
```
python -m chatbot.main
```

## Considerations

This chatbot has been developed over a Langchain Agent in which the RAG and the REPL are tools, intead of using a Chain, and by default will respond using the LLM itself. Currently the selection of a tool by the Agent depends exclusively on the description of each tool. 

The vector store used is a built in Chroma, instead of an standalone vector store like Milvus, Elasticsearch, etc. for development speed sake.

The project doesn't have loggins