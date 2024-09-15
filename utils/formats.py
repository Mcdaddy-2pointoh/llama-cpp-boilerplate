# Import 
from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional

# OpenAI message format
class OpenAIMessage(BaseModel):
    role: Literal['system', 'user', 'assistant'] = Field(
        ..., description="Role of the message sender. It must be one of 'system', 'user', or 'assistant'."
    )
    content: str = Field(
        ..., description="The content of the message, which must be non-empty."
    )
    name: Optional[str] = Field(
        None, description="Optional name of the user or entity sending the message."
    )

    @field_validator('content')
    def content_cannot_be_empty(cls, value):
        if not value.strip():
            raise ValueError("Content cannot be empty or just whitespace.")
        return value

# OpenAI Chat messages format
class OpenAIChatMessages(BaseModel):
    messages: List[OpenAIMessage] = Field(
        ..., description="A list of messages following OpenAI's message format."
    )

    @field_validator('messages')
    def messages_must_not_be_empty(cls, value):
        if len(value) == 0:
            raise ValueError("The list of messages cannot be empty.")
        return value

    class Config:
        schema_extra = {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "How can I improve my Python skills?", "name": "John"},
                    {"role": "assistant", "content": "You can practice coding daily and work on projects."}
                ]
            }
        }

