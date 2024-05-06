# Summary Buffer Memory

from chatbot.modules.memory.common import MEMORY_KEY, INPUT_KEY
from chatbot.modules.llm import llm as default_llm
from chatbot.modules.memory.message_history import ChatMessageHistory

from langchain.memory import ConversationSummaryBufferMemory

import os


class ConversationSummaryBufferMemoryFactory:
    # Currently doesn't difference between users neither has persistent memory
    def get_instance(llm=None, old_history_summary=""):
        if not llm:
            llm = default_llm
            
        return ConversationSummaryBufferMemory(
            llm=llm,
            memory_key=MEMORY_KEY,
            input_key=INPUT_KEY,
            chat_memory=ChatMessageHistory(),
            moving_summary_buffer=old_history_summary,
            max_token_limit=os.getenv("MAX_TOKEN_LIMIT"),
        )
