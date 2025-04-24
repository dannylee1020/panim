from pydantic import BaseModel

class GenerateRequest(BaseModel):
    instruction: str

class GenerateResponse(BaseModel):
    answer: str
    error: str | None = None