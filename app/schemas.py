# app/schemas.py

from typing import List, Dict, Any
from pydantic import BaseModel


# ===== 普通文本生成 =====

class TextGenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.8
    top_p: float = 0.95


class TextGenerationResponse(BaseModel):
    prompt: str
    generated: str


# ===== 带格式要求的文本生成（RL 策略选择模板） =====

class RLTextGenerationRequest(BaseModel):
    prompt: str


class RLTextGenerationResponse(BaseModel):
    prompt: str
    chosen_template_index: int
    formatted_prompt: str
    generated: str
    reward: float
    info: Dict[str, Any]


# ===== RL 训练请求/响应 =====

class RLTrainRequest(BaseModel):
    """
    简单的 REINFORCE 训练接口：
    - prompts: 一批用来训练的 prompt
    - episodes_per_prompt: 每个 prompt 跑多少个 episode
    """
    prompts: List[str]
    episodes_per_prompt: int = 1


class RLTrainResponse(BaseModel):
    avg_reward: float
    loss: float
    num_episodes: int
