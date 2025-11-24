# sample_images.py
import torch

from helper_lib.model import get_model
from helper_lib.generator import (
    generate_diffusion_samples,
    generate_energy_samples,
)


def select_device():
    return (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


if __name__ == "__main__":
    device = select_device()
    print("Using device:", device)

    # ========= 1. 加载 Diffusion 模型并生成样本 =========
    diffusion_ckpt = "./artifacts_diffusion/diffusion_ep002.pt"  # 如果你只有 ep001 就改成 001
    print(f"Loading diffusion checkpoint from {diffusion_ckpt}")

    unet = get_model("diffusion", image_size=64, num_channels=3)
    unet.load_state_dict(torch.load(diffusion_ckpt, map_location=device))

    generate_diffusion_samples(
        unet,
        device=device,
        num_samples=16,        # 4x4 网格
        diffusion_steps=50,    # 反向扩散步数
        image_size=64,
        num_channels=3,
    )

    # ========= 2. 加载 Energy Model 并生成样本 =========
    energy_ckpt = "./artifacts_energy/energy_ep002.pt"
    print(f"Loading energy checkpoint from {energy_ckpt}")

    ebm = get_model("energy")
    ebm.load_state_dict(torch.load(energy_ckpt, map_location=device))

    generate_energy_samples(
        ebm,
        device=device,
        num_samples=16,   # 4x4 网格
        steps=60,         # Langevin 更新步数
        step_size=10.0,
        noise_std=0.01,
        image_size=64,
        num_channels=3,
    )
