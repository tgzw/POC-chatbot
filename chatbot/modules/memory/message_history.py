from chatbot.database.database import DatabaseController

from typing import Sequence
from langchain.schema import messages_from_dict, messages_to_dict
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage


class ChatMessageHistory(BaseChatMessageHistory):
    def __init__(self) -> None:
        super().__init__()
        self.db_controller = DatabaseController()

    @property
    def messages(self):
        return messages_from_dict(self.db_controller.get_message_history())

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        serialized_messages = messages_to_dict(messages)
        self.db_controller.save_messages(serialized_messages)

    def clear(self):
        pass
