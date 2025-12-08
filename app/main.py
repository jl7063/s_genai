# app/main.py

from typing import Optional, List

import base64
import io
import math
import os

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

import torch
from torchvision import transforms
from torchvision.utils import save_image

from helper_lib.model import get_model
from helper_lib.generator import generate_diffusion_samples, generate_energy_samples

# ====== RL + LLM 相关 ======
from .llm_wrapper import get_llm
from .rl_env import run_single_episode
from .rl_policy import get_agent, Trajectory
from .schemas import (
    TextGenerationRequest,
    TextGenerationResponse,
    RLTextGenerationRequest,
    RLTextGenerationResponse,
    RLTrainRequest,
    RLTrainResponse,
)


# ---------- 设备选择 ----------

try:
    # 优先用你在 helper_lib.trainer 里写好的 select_device
    from helper_lib.trainer import select_device
except Exception:
    # 兜底：mps -> cuda -> cpu
    def select_device():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


DEVICE = select_device()

# ---------- 图像分类的预处理配置 ----------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

TRANSFORM = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
)

app = FastAPI(
    title="Multi-Modal API: CNN + GAN + Diffusion + Energy + RL-LLM",
    description=(
        "原有的图像分类 + 生成 API（Module 3/6/8），"
        "以及用于文本生成的 GPT-2 + RL 格式控制 API（Module 9-11）。"
    ),
)

# ---------- 全局模型句柄 ----------

MODEL = None  # classifier
LABELS: Optional[List[str]] = None

GAN = None  # wrapper with generator/discriminator
GAN_Z_DIM = 100

DIFFUSION_MODEL = None  # UNet for diffusion
ENERGY_MODEL = None  # Energy-based model


def _load_labels(path: str) -> List[str]:
    with open(path, "r") as f:
        return [x.strip() for x in f if x.strip()]


# ---------- 启动时加载所有模型 ----------

@app.on_event("startup")
def _warmup():
    """Load classifier, GAN, Diffusion, Energy models on startup."""
    global MODEL, LABELS, GAN, DEVICE, DIFFUSION_MODEL, ENERGY_MODEL

    DEVICE = select_device()
    print(f"[startup] device = {DEVICE}")

    # ---- Load classifier ----
    try:
        LABELS = _load_labels("./artifacts/labels.txt")
        num_classes = len(LABELS)
        print(f"[startup] labels loaded: {num_classes} classes")

        MODEL = get_model("assignmentcnn", num_classes=num_classes)
        MODEL.load_state_dict(torch.load("./artifacts/model.pt", map_location="cpu"))
        MODEL.to(DEVICE).eval()

        print("[startup] Classifier model & labels loaded.")
    except Exception as e:
        import traceback

        print("[startup] Classifier warmup failed:", e)
        traceback.print_exc()
        MODEL = None
        LABELS = None

    # ---- Load GAN generator (optional) ----
    try:
        if os.path.exists("./artifacts/gan_generator.pt"):
            GAN = get_model("gan", z_dim=GAN_Z_DIM)
            state = torch.load("./artifacts/gan_generator.pt", map_location=DEVICE)
            GAN.generator.load_state_dict(state)
            GAN.generator.to(DEVICE).eval()
            print("[startup] GAN generator loaded from ./artifacts/gan_generator.pt")
        else:
            GAN = get_model("gan", z_dim=GAN_Z_DIM)  # random init for API smoke test
            GAN.generator.to(DEVICE).eval()
            print(
                "[startup] No gan_generator.pt found. "
                "Using randomly initialized generator."
            )
    except Exception as e:
        import traceback

        print("[startup] GAN warmup failed:", e)
        traceback.print_exc()
        GAN = None  # disable GAN endpoint if loading fails

    # ---- Load Diffusion model ----
    try:
        ckpt_path = "./artifacts_diffusion/diffusion_ep002.pt"
        if not os.path.exists(ckpt_path):
            # 如果你只有 ep001，就自动退回 001
            fallback = "./artifacts_diffusion/diffusion_ep001.pt"
            if os.path.exists(fallback):
                ckpt_path = fallback

        DIFFUSION_MODEL = get_model("diffusion", image_size=64, num_channels=3)
        DIFFUSION_MODEL.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE),
        )
        DIFFUSION_MODEL.to(DEVICE).eval()
        print(f"[startup] Diffusion model loaded from {ckpt_path}")
    except Exception as e:
        import traceback

        print("[startup] Diffusion warmup failed:", e)
        traceback.print_exc()
        DIFFUSION_MODEL = None

    # ---- Load Energy model ----
    try:
        ckpt_path = "./artifacts_energy/energy_ep002.pt"
        if not os.path.exists(ckpt_path):
            fallback = "./artifacts_energy/energy_ep001.pt"
            if os.path.exists(fallback):
                ckpt_path = fallback

        ENERGY_MODEL = get_model("energy")
        ENERGY_MODEL.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        ENERGY_MODEL.to(DEVICE).eval()
        print(f"[startup] Energy model loaded from {ckpt_path}")
    except Exception as e:
        import traceback

        print("[startup] Energy warmup failed:", e)
        traceback.print_exc()
        ENERGY_MODEL = None

    # ---- 预热 LLM / RL Agent（可选，避免第一次调用太慢） ----
    try:
        _ = get_llm()
        _ = get_agent()
        print("[startup] LLM (GPT-2) and RL agent initialized.")
    except Exception as e:
        print("[startup] LLM/RL warmup failed:", e)


# ---------- 图像分类 Endpoint ----------


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """使用你在 Module 3 里训练的 CNN 对上传图片做分类。"""
    try:
        content = await file.read()
        img = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    if MODEL is None or LABELS is None:
        raise HTTPException(status_code=500, detail="Classifier model not loaded")

    try:
        x = TRANSFORM(img).unsqueeze(0).to(DEVICE)  # [1,3,64,64]
        with torch.no_grad():
            logits = MODEL(x)
            prob = torch.softmax(logits, dim=1)
            conf, idx = prob.max(dim=1)
            idx = idx.item()

        if idx < 0 or idx >= len(LABELS):
            raise IndexError(
                f"class index {idx} out of range for labels size {len(LABELS)}"
            )

        return JSONResponse(
            {"class": LABELS[idx], "confidence": float(conf.item())},
        )
    except Exception as e:
        import traceback

        print("[/classify] inference error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


# ---------- GAN Generation ----------


class GanRequest(BaseModel):
    num_samples: int = 16  # best as perfect squares: 4, 9, 16, 25, ...
    z_dim: Optional[int] = None  # override z-dim per request (optional)


@app.post("/generate_gan")
def generate_gan(req: GanRequest):
    """
    Generate an MNIST grid (Base64 PNG).

    - If ./artifacts/gan_generator.pt is present, it uses that trained generator.
    - Otherwise it uses a randomly initialized generator (good for API testing).
    """
    if GAN is None:
        raise HTTPException(status_code=500, detail="GAN is not available")

    # If a different z_dim is requested, build a temporary generator
    z_dim = int(req.z_dim) if req.z_dim else GAN_Z_DIM

    if z_dim != getattr(GAN, "z_dim", GAN_Z_DIM):
        try:
            _tmp_gan = get_model("gan", z_dim=z_dim)
            _tmp_gan.generator.to(DEVICE).eval()
            generator = _tmp_gan.generator
        except Exception as e:
            print("[/generate_gan] fallback to existing GAN due to:", e)
            generator = GAN.generator
            z_dim = getattr(GAN, "z_dim", GAN_Z_DIM)
    else:
        generator = GAN.generator

    num = max(1, int(req.num_samples))
    nrow = int(math.sqrt(num))
    if nrow * nrow != num:
        nrow = max(1, int(round(math.sqrt(num))))

    with torch.no_grad():
        z = torch.randn(num, z_dim, device=DEVICE)
        imgs = generator(z).cpu()  # Tanh output in [-1, 1]
        imgs = (imgs + 1) / 2.0  # map back to [0, 1]

    # Encode as Base64 PNG
    buffer = io.BytesIO()
    save_image(imgs, buffer, nrow=nrow, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return {
        "num_samples": num,
        "z_dim": z_dim,
        "image_base64_png": b64,
    }


# ---------- Diffusion Generation ----------


class DiffusionRequest(BaseModel):
    num_samples: int = 16
    diffusion_steps: int = 50


@app.post("/generate_diffusion")
def generate_diffusion(req: DiffusionRequest):
    """
    使用训练好的 Diffusion 模型生成 CIFAR10 风格图像。

    为了让 API 快速返回，这里强制限制为最多 4 张图、10 个反向扩散步。
    """
    if DIFFUSION_MODEL is None:
        raise HTTPException(status_code=500, detail="Diffusion model not loaded")

    # 限制规模，避免接口计算时间太长
    num_samples = min(max(1, req.num_samples), 4)
    diffusion_steps = min(max(1, req.diffusion_steps), 10)

    imgs = generate_diffusion_samples(
        DIFFUSION_MODEL,
        device=DEVICE,
        num_samples=num_samples,
        diffusion_steps=diffusion_steps,
        image_size=64,
        num_channels=3,
        show=False,  # API 下不弹窗
    )

    # 这里只返回一些简单的统计信息，说明生成成功
    return {
        "status": "ok",
        "detail": "Diffusion samples generated.",
        "num_samples": int(imgs.shape[0]),
        "shape": list(imgs.shape),
    }


# ---------- Energy-Based Model Generation ----------


class EnergyRequest(BaseModel):
    num_samples: int = 16
    steps: int = 60
    step_size: float = 10.0
    noise_std: float = 0.01


@app.post("/generate_energy")
def generate_energy(req: EnergyRequest):
    """
    使用训练好的 Energy-Based Model 通过 Langevin dynamics 对输入图像
    做梯度下降，生成低能量样本。
    """
    if ENERGY_MODEL is None:
        raise HTTPException(status_code=500, detail="Energy model not loaded")

    num_samples = min(max(1, req.num_samples), 4)
    steps = min(max(1, req.steps), 20)

    imgs = generate_energy_samples(
        ENERGY_MODEL,
        device=DEVICE,
        num_samples=num_samples,
        steps=steps,
        step_size=req.step_size,
        noise_std=req.noise_std,
        image_size=64,
        num_channels=3,
        show=False,
    )

    return {
        "status": "ok",
        "detail": "Energy-based samples generated.",
        "num_samples": int(imgs.shape[0]),
        "shape": list(imgs.shape),
    }


# ========== 新增：LLM 文本生成 + RL 格式控制 ==========


@app.get("/health")
def health():
    """简单健康检查，老师 / 自己测 Docker 时非常好用。"""
    return {"status": "ok"}


@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(req: TextGenerationRequest):
    """
    使用（微调后的）GPT-2 做普通文本生成。

    这是对 Modules 3 & 7 里 “text generation API” 的升级：
    背后模型变成了在 Module 9 上 fine-tune 过的 GPT-2（如果存在），
    否则回退到 openai-community/gpt2。
    """
    llm = get_llm()
    generated = llm.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return TextGenerationResponse(prompt=req.prompt, generated=generated)


@app.post("/generate_formatted", response_model=RLTextGenerationResponse)
async def generate_formatted(req: RLTextGenerationRequest):
    """
    使用 RL 策略网络选择一个模板（action），
    然后用 GPT-2 生成符合指定格式（Thought / Answer）的回答。

    这就是 Module 10 / 11 要求的 “post-training to enforce a given format”。
    """
    agent = get_agent()
    episode = agent.generate_with_policy(req.prompt)

    return RLTextGenerationResponse(
        prompt=episode.prompt,
        chosen_template_index=episode.template_index,
        formatted_prompt=episode.formatted_prompt,
        generated=episode.generated,
        reward=episode.reward,
        info=episode.info,
    )


@app.post("/rl/train_once", response_model=RLTrainResponse)
async def rl_train_once(req: RLTrainRequest):
    """
    使用 REINFORCE 对策略网络做一次简单的训练。

    - 对每个 prompt 采样 episodes_per_prompt 次
    - 每个 episode:
        * 策略选择一个模板（action）
        * 用该模板调用 GPT-2 生成文本
        * 根据格式奖励函数得到 reward
    - 优化目标：最大化期望 reward
    """
    agent = get_agent()

    trajectories: List[Trajectory] = []
    rewards: List[float] = []

    for prompt in req.prompts:
        for _ in range(req.episodes_per_prompt):
            action, _ = agent.select_action(prompt)
            episode = run_single_episode(prompt, action)

            traj = Trajectory(
                prompts=[prompt],
                actions=[action],
                rewards=[episode.reward],
                episode_results=[episode],
            )
            trajectories.append(traj)
            rewards.append(episode.reward)

    loss = agent.reinforce_update(trajectories)
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0

    # 尝试把策略权重保存下来（老师不一定会用到，但体现完整流程）
    try:
        os.makedirs("artifacts_rl", exist_ok=True)
        agent.save("artifacts_rl/rl_policy.pt")
    except Exception as e:
        print("[/rl/train_once] Warning: failed to save RL policy:", e)

    return RLTrainResponse(
        avg_reward=avg_reward,
        loss=loss,
        num_episodes=len(rewards),
    )


# ---------- Root ----------


@app.get("/")
def root():
    return {
        "status": "ok",
        "use": {
            "classify": "POST /classify (multipart/form-data: file=)",
            "gan": "POST /generate_gan (json: {num_samples: 16, z_dim: 100})",
            "diffusion": "POST /generate_diffusion (json: {num_samples: 16, diffusion_steps: 50})",
            "energy": "POST /generate_energy (json: {num_samples: 16, steps: 60, step_size: 10.0, noise_std: 0.01})",
            "text_generate": "POST /generate (json: {prompt, max_new_tokens, temperature, top_p})",
            "text_generate_formatted": "POST /generate_formatted (json: {prompt})",
            "rl_train": "POST /rl/train_once (json: {prompts: [...], episodes_per_prompt: 1})",
        },
        "docs": "/docs",
    }
