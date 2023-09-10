from pydantic import BaseModel
from typing import Union

class ChatCompletionRequest(BaseModel):
    system: Union[str, None]
    user: str
    temperature: float = 0.8

class ChatCompletionResponse(BaseModel):
    response: str