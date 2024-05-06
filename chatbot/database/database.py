# Placeholder of future database logic


# "Database" of messages
MESSAGE_HISTORY = [
    {
        "type": "human",
        "data": {
            "content": "Hey how are you? In what can you help me?",
            "additional_kwargs": {},
            "response_metadata": {},
            "type": "human",
            "name": None,
            "id": None,
            "example": False,
        },
    },
    {
        "type": "ai",
        "data": {
            "content": "Hello! I'm here to assist you with a wide range of tasks. Feel free to ask me anything or let me know how I can help you.",
            "additional_kwargs": {},
            "response_metadata": {},
            "type": "ai",
            "name": None,
            "id": None,
            "example": False,
            "tool_calls": [],
            "invalid_tool_calls": [],
        },
    },
]


class DatabaseSessionManager:
    def __init__(self) -> None:
        pass

    def connect(self):
        pass

    def close(self):
        pass


class DatabaseController:
    def __init__(self) -> None:
        self.db = DatabaseSessionManager()

    def get_message_history(self):
        return MESSAGE_HISTORY

    def save_messages(self, messages):
        MESSAGE_HISTORY.extend(messages)
