# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .llm_wrapper import get_llm
from .schemas import (
    GenerateRequest,
    GenerateResponse,
    GenerateFormattedResponse,
    RLTrainRequest,
    RLTrainResponse,
)
from .rl_policy import get_agent, Trajectory
from .rl_env import run_single_episode


app = FastAPI(title="RL-Formatted Text API")

# 可选：允许本地前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    """
    原始 LLM 生成，不加 RL。
    """
    llm = get_llm()
    text = llm.generate(req.prompt, max_new_tokens=req.max_new_tokens)
    return GenerateResponse(prompt=req.prompt, generated=text)


@app.post("/generate_formatted", response_model=GenerateFormattedResponse)
def generate_formatted(req: GenerateRequest):
    """
    使用 RL 策略网络选择模板，再调用 LLM 生成。
    这是本作业最关键的 API 之一。
    """
    agent = get_agent()
    episode = agent.generate_with_policy(req.prompt)

    return GenerateFormattedResponse(
        prompt=req.prompt,
        chosen_template_index=episode.template_index,
        generated=episode.generated,
        reward=episode.reward,
        info=episode.info,
    )


@app.post("/rl/train_once", response_model=RLTrainResponse)
def rl_train_once(req: RLTrainRequest):
    """
    简单的 RL 训练端点：老师在他的机器上跑 Docker，hit 这个 API
    就可以看到你 RL training 的效果。
    """
    agent = get_agent()
    prompts = [
        "Explain reinforcement learning in one paragraph.",
        "What is the difference between Q-learning and policy gradients?",
        "Why might we use a discount factor in RL?",
        "Give a short explanation of the exploration-exploitation tradeoff.",
    ]

    total_reward = 0.0
    total_steps = req.epochs * req.episodes_per_epoch

    for _ in range(req.epochs):
        trajectories = []
        for i in range(req.episodes_per_epoch):
            prompt = prompts[i % len(prompts)]
            action, _ = agent.select_action(prompt)
            episode = run_single_episode(prompt, action)

            traj = Trajectory(
                prompts=[prompt],
                actions=[action],
                rewards=[episode.reward],
                episode_results=[episode],
            )
            trajectories.append(traj)
            total_reward += episode.reward

        agent.reinforce_update(trajectories)

    final_avg_reward = total_reward / total_steps if total_steps > 0 else 0.0

    return RLTrainResponse(
        epochs=req.epochs,
        episodes_per_epoch=req.episodes_per_epoch,
        final_avg_reward=final_avg_reward,
    )
