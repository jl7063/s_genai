# app/llm_wrapper.py
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLMWrapper:
    """
    使用 HuggingFace GPT-2 的简单封装。
    后面 FastAPI 和 RL 都通过这个类来生成文本。
    """

    def __init__(
        self,
        model_name_or_path: str = "openai-community/gpt2",
        device: Optional[str] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        # 加载 tokenizer 和 model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path
        ).to(self.device)
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

        # 简单截断：如果生成里重复了 prompt，就只取后面部分
        if text.startswith(prompt):
            text = text[len(prompt):]

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
