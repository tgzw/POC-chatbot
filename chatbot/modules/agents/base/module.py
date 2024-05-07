from chatbot.modules.agents.base.prompts import PREFIX, FORMAT_INSTRUCTIONS, SUFFIX
from chatbot.modules.llm import llm
from chatbot.modules.tools import retriever_tool, repl_tool
from chatbot.modules.memory.summary_buffer_memory import (
    ConversationSummaryBufferMemoryFactory,
)

from langchain.agents.conversational.base import ConversationalAgent
from langchain.agents import AgentExecutor


base_agent = ConversationalAgent
tools = [retriever_tool, repl_tool]
memory = ConversationSummaryBufferMemoryFactory.get_instance()

agent = base_agent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    format_instructions=FORMAT_INSTRUCTIONS,
    prefix=PREFIX,
    suffix=SUFFIX,
)

agent_executor = AgentExecutor(
    agent=agent, 
    memory=memory, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True)
