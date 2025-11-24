# train_diffusion_main.py
import torch

from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    # 注意：image_size=64，因为 data_loader 里 Resize(64)
    model = get_model("diffusion", image_size=64, num_channels=3)

    train_diffusion(
        model,
        epochs=5,             # 可以先跑 1~2 epoch 测试
        lr=1e-4,
        device=device,
        batch_size=64,
        out_dir="./artifacts_diffusion",
    )
