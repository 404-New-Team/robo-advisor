from pydantic import BaseModel
from typing import Any


class ErrorResponse(BaseModel):
    status: str = "error"
    code: int
    message: str
    detail: str | None = None
