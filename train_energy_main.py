# train_energy_main.py
import torch

from helper_lib.model import get_model
from helper_lib.trainer import train_energy_model

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    model = get_model("energy")

    train_energy_model(
        model,
        epochs=2,              # 先跑 2 个 epoch 测试，OK 再改大
        lr=1e-4,
        device=device,
        batch_size=64,
        out_dir="./artifacts_energy",
    )
