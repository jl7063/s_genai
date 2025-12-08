# app/schemas.py
from typing import Dict, Any

from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128


class GenerateResponse(BaseModel):
    prompt: str
    generated: str


class GenerateFormattedResponse(BaseModel):
    prompt: str
    chosen_template_index: int
    generated: str
    reward: float
    info: Dict[str, Any]


class RLTrainRequest(BaseModel):
    epochs: int = 5
    episodes_per_epoch: int = 8


class RLTrainResponse(BaseModel):
    epochs: int
    episodes_per_epoch: int
    final_avg_reward: float
