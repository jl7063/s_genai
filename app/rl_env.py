# app/rl_env.py
import re
from dataclasses import dataclass
from typing import List, Dict, Any

from .llm_wrapper import get_llm


# 动作空间：我们允许策略从这些模板里选一个
TEMPLATES: List[str] = [
    # 模板 0：最普通的，不强制格式（baseline）
    "You are a helpful assistant.\nUser question: {prompt}\nAssistant:",
    # 模板 1：要求 Thought 和 Answer 格式
    (
        "You are a careful assistant. For every answer, "
        "first think step by step under a line starting with 'Thought:', "
        "then give the final result under a line starting with 'Answer:'.\n\n"
        "User question: {prompt}\nAssistant:"
    ),
    # 模板 2：另一种更明确的格式要求
    (
        "Answer the following question using exactly two sections:\n"
        "Thought: ...\nAnswer: ...\n\nQuestion: {prompt}\nAssistant:"
    ),
]


@dataclass
class EpisodeResult:
    prompt: str
    template_index: int
    formatted_prompt: str
    generated: str
    reward: float
    info: Dict[str, Any]


def apply_template(prompt: str, template_index: int) -> str:
    template = TEMPLATES[template_index]
    return template.format(prompt=prompt)


def format_reward(generated_text: str) -> (float, Dict[str, Any]):
    """
    根据生成文本是否符合 'Thought:' + 'Answer:' 格式给予奖励。
    你可以根据老师给的格式要求调整规则。
    """

    has_thought = bool(re.search(r"Thought\s*:", generated_text))
    has_answer = bool(re.search(r"Answer\s*:", generated_text))

    reward = 0.0

    # 同时有 Thought 和 Answer -> 大奖
    if has_thought and has_answer:
        reward += 1.0
    # 只有一个 -> 小奖
    elif has_thought or has_answer:
        reward += 0.3

    # 长度奖励/惩罚（太短/太长都不好）
    length = len(generated_text.split())
    if 30 <= length <= 300:
        reward += 0.2
    elif length < 10:
        reward -= 0.2

    info = {
        "has_thought": has_thought,
        "has_answer": has_answer,
        "length": length,
    }
    return reward, info


def run_single_episode(prompt: str, template_index: int) -> EpisodeResult:
    """
    环境里的一次 Episode：
    - 策略选择一个模板 index（action）
    - 我们用这个模板 + prompt 调用 LLM
    - 用奖励函数评估格式
    """
    formatted_prompt = apply_template(prompt, template_index)

    llm = get_llm()
    generated = llm.generate(formatted_prompt)

    reward, info = format_reward(generated)

    return EpisodeResult(
        prompt=prompt,
        template_index=template_index,
        formatted_prompt=formatted_prompt,
        generated=generated,
        reward=reward,
        info=info,
    )
