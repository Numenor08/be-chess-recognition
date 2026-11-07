from pydantic import BaseModel, Field


class AnalyzeFenRequest(BaseModel):
    fen: str
    depth: int = Field(default=12, ge=1, le=30)
    show_variations: int = Field(default=3, ge=1, le=5)
