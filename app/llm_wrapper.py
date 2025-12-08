# app/llm_wrapper.py

from typing import Optional, List
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMWrapper:
    """
    使用 HuggingFace GPT-2 的简单封装。

    优先尝试加载你在 Module 9 微调好的本地权重（例如 SQuAD 上 fine-tune 的 GPT-2），
    路径默认假设为 ./artifacts_gpt2。
    如果本地没有微调权重，则回退到公开的 "openai-community/gpt2" 预训练模型。

    后面 FastAPI 和 RL 都通过这个类来生成文本。
    """

    def __init__(
        self,
        finetuned_path: str = "./artifacts_gpt2",
        base_model_name: str = "openai-community/gpt2",
        device: Optional[str] = None,
    ):
        # 设备选择
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # 如果本地有你在 Module 9 微调好的 GPT-2，就优先用本地的
        if os.path.isdir(finetuned_path):
            model_source = finetuned_path
        else:
            model_source = base_model_name

        # 加载 tokenizer 和 model
        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_source).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_tokens: Optional[List[str]] = None,
    ) -> str:
        """根据给定 prompt 生成文本，并做简单截断。"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        text = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )

        # 如果生成结果里重复了 prompt，就只保留后半段
        if text.startswith(prompt):
            text = text[len(prompt):]

        # 简单 stop token 截断
        if stop_tokens:
            for st in stop_tokens:
                if st in text:
                    text = text.split(st)[0]
                    break

        return text.strip()


# ===== 全局单例：FastAPI / RL 共用 =====
_llm_instance: Optional[LLMWrapper] = None


def get_llm() -> LLMWrapper:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMWrapper()
    return _llm_instance
