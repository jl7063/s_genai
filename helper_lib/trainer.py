# helper_lib/trainer.py
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

# --- For classifier training (your current pipeline) ---
from .data_loader import get_data_loader  # make sure it resizes to 64 if your model expects 64x64

# --- For GAN / MNIST ---
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


# -----------------------------
# Device selection (MPS > CUDA > CPU)
# -----------------------------
def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# =========================================================
# 1) Classifier training (kept compatible with your code)
# =========================================================
def train_model(
    model: nn.Module,
    epochs: int = 12,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    out_dir: str = "./artifacts",
    batch_size: int = 64,
):
    """
    Supervised classifier training loop (CrossEntropy).
    Uses your project get_data_loader(train=...), and saves best weights/labels.
    """
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = select_device()
    else:
        device = torch.device(device)

    train_loader = get_data_loader(train=True,  batch_size=batch_size)
    val_loader   = get_data_loader(train=False, batch_size=batch_size)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    for ep in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total, correct, tot_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * y.size(0)
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()

        tr_loss = tot_loss / max(1, total)
        tr_acc = correct / max(1, total)

        # ---- Eval ----
        model.eval()
        v_tot, v_hit = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                v_tot += y.size(0)
                v_hit += (pred == y).sum().item()
        val_acc = v_hit / max(1, v_tot)

        print(f"Epoch {ep}/{epochs}  loss={tr_loss:.4f}  train_acc={tr_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{out_dir}/model.pt")

    # CIFAR-10 default labels (edit if your dataset differs)
    labels = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    with open(f"{out_dir}/labels.txt", "w") as f:
        f.write("\n".join(labels))

    print(f"Best val_acc={best_acc:.3f}. Weights saved to {out_dir}/model.pt")


# =========================================================
# 2) GAN training for MNIST (Vanilla GAN with BCE loss)
#    - Matches the GAN model you added in helper_lib/model.py
#    - Normalizes MNIST to [-1, 1] to match generator Tanh output
# =========================================================
@torch.no_grad()
def _randn_z(batch: int, z_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch, z_dim, device=device)


def _build_mnist_loader(batch_size: int, num_workers: int = 2) -> DataLoader:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # to [-1, 1], matches generator Tanh
    ])
    ds = datasets.MNIST(root="./data", train=True, transform=tfm, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


def generate_gan_samples(
    model: nn.Module,
    num_samples: int = 16,
    out_dir: str = "./artifacts_gan",
    filename: str = "gan_grid.png",
    device: Optional[torch.device] = None,
) -> str:
    """
    Generate and save a grid of samples from a trained (or current) GAN.
    Returns the saved file path.
    """
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = select_device()

    model.eval()
    model.generator.to(device)
    with torch.no_grad():
        z = _randn_z(num_samples, model.z_dim, device=device)
        imgs = model.generator(z).cpu()
        imgs = (imgs + 1) / 2.0  # back to [0,1]

    path = os.path.join(out_dir, filename)
    save_image(imgs, path, nrow=int(num_samples ** 0.5))
    return path


def train_gan(
    model: nn.Module,
    epochs: int = 5,
    lr: float = 2e-4,
    beta1: float = 0.5,
    batch_size: int = 128,
    out_dir: str = "./artifacts_gan",
    device: Optional[str] = None,
    num_workers: int = 2,
    sample_every: int = 1,
):
    """
    Vanilla GAN training loop on MNIST.
    - Discriminator: BCE(real=1, fake=0)
    - Generator: tries to make D(fake) -> 1
    Saves sample grids every `sample_every` epochs and final weights.
    """
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = select_device()
    else:
        device = torch.device(device)

    # Move submodules
    model.generator.to(device)
    model.discriminator.to(device)
    model.train()

    # Data
    loader = _build_mnist_loader(batch_size=batch_size, num_workers=num_workers)

    # Loss/opt
    bce = nn.BCELoss()
    opt_g = optim.Adam(model.generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_d = optim.Adam(model.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    z_dim = model.z_dim

    for ep in range(1, epochs + 1):
        pbar = tqdm(loader, desc=f"[GAN] Epoch {ep}/{epochs}")
        for real, _ in pbar:
            real = real.to(device)                     # [B,1,28,28]
            B = real.size(0)
            y_real = torch.ones(B, 1, device=device)   # label 1 for real
            y_fake = torch.zeros(B, 1, device=device)  # label 0 for fake

            # ---- Train D ----
            opt_d.zero_grad(set_to_none=True)
            d_real = model.discriminator(real)
            loss_d_real = bce(d_real, y_real)

            z = _randn_z(B, z_dim, device=device)
            fake = model.generator(z)
            d_fake = model.discriminator(fake.detach())
            loss_d_fake = bce(d_fake, y_fake)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            opt_d.step()

            # ---- Train G ----
            opt_g.zero_grad(set_to_none=True)
            z = _randn_z(B, z_dim, device=device)
            fake = model.generator(z)
            d_fake_for_g = model.discriminator(fake)
            loss_g = bce(d_fake_for_g, y_real)  # make fake look real
            loss_g.backward()
            opt_g.step()

            pbar.set_postfix(loss_d=float(loss_d), loss_g=float(loss_g))

        # Save samples each epoch (or every N epochs)
        if sample_every and (ep % sample_every == 0):
            grid_path = generate_gan_samples(
                model,
                num_samples=16,
                out_dir=out_dir,
                filename=f"epoch_{ep:03d}.png",
                device=device,
            )
            print(f"[GAN] Saved samples to {grid_path}")

    # Save final weights
    torch.save(model.generator.state_dict(), os.path.join(out_dir, "G.pt"))
    torch.save(model.discriminator.state_dict(), os.path.join(out_dir, "D.pt"))
    print(f"[GAN] Training complete. Weights saved to {out_dir}/G.pt and {out_dir}/D.pt")


# =========================================================
# 3) Diffusion training: UNet learns to predict noise
#    - Works with the UNet you added in helper_lib/model.py
# =========================================================

def diffusion_schedule(
    diffusion_times: torch.Tensor,
    min_signal_rate: float = 0.02,
    max_signal_rate: float = 0.95,
):
    """
    Offset-cosine diffusion schedule (和 Module 8 practical 一致的思路):
      - 输入: diffusion_times in [0, 1], shape (B, 1, 1, 1)
      - 输出:
          noise_rates  shape (B, 1, 1, 1)
          signal_rates shape (B, 1, 1, 1)
    """
    device = diffusion_times.device
    start_angle = torch.arccos(torch.tensor(max_signal_rate, device=device))
    end_angle = torch.arccos(torch.tensor(min_signal_rate, device=device))
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    signal_rates = torch.cos(diffusion_angles)
    noise_rates = torch.sin(diffusion_angles)
    return noise_rates, signal_rates


def train_diffusion(
    model: nn.Module,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[str] = None,
    batch_size: int = 64,
    out_dir: str = "./artifacts_diffusion",
):
    """
    训练 Diffusion 的 UNet，让它学会在任意噪声强度下预测加到图像上的 noise。

    使用方式（之后我们会写 main / API 调用）：
        from helper_lib.model import get_model
        from helper_lib.trainer import train_diffusion

        unet = get_model("diffusion", image_size=64, num_channels=3)
        train_diffusion(unet, epochs=10, batch_size=64)

    训练过程：
      1. 从 CIFAR-10 取一批干净图像 x (归一化到 [-1,1])
      2. 随机采样 diffusion_times ~ U(0,1)
      3. 用 schedule 得到 noise_rates, signal_rates
      4. 采样 eps ~ N(0, I)，构造 noisy_images = signal * x + noise * eps
      5. UNet(noisy_images, noise_rates) 预测 eps_hat
      6. 用 MSE( eps_hat, eps ) 做 loss，反向传播
    """
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = select_device()
    else:
        device = torch.device(device)

    model = model.to(device)
    model.train()

    train_loader = get_data_loader(train=True, batch_size=batch_size)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"[Diffusion] Epoch {ep}/{epochs}")
        for clean_images, _ in pbar:
            clean_images = clean_images.to(device)  # (B, 3, 64, 64)，已归一化到 [-1,1]
            B = clean_images.size(0)
            total_samples += B

            # 1) 随机采样时间 t ~ Uniform(0,1)
            diffusion_times = torch.rand(B, 1, 1, 1, device=device)

            # 2) 根据 schedule 计算 noise_rates 和 signal_rates
            noise_rates, signal_rates = diffusion_schedule(diffusion_times)

            # 3) 采样真实噪声 eps
            noises = torch.randn_like(clean_images)

            # 4) 构造带噪图像 x_t
            noisy_images = signal_rates * clean_images + noise_rates * noises

            # 5) UNet 预测噪声
            optimizer.zero_grad(set_to_none=True)
            pred_noises = model(noisy_images, noise_rates)

            loss = criterion(pred_noises, noises)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            avg_loss = total_loss / max(1, total_samples)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        print(f"[Diffusion] Epoch {ep}/{epochs}  loss={avg_loss:.4f}")

        # 每个 epoch 保存一次权重，方便之后加载生成
        ckpt_path = os.path.join(out_dir, f"diffusion_ep{ep:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Diffusion] Saved checkpoint to {ckpt_path}")

    print(f"[Diffusion] Training complete. Last weights in {ckpt_path}")
    
def train_energy_model(
    model: nn.Module,
    epochs: int = 10,
    lr: float = 1e-4,
    device: Optional[str] = None,
    batch_size: int = 64,
    out_dir: str = "./artifacts_energy",
):
    """
    训练 Energy Model (简单对比损失)，使用 CIFAR-10。

    Loss = E(real) - E(fake)
      - real: 真实 CIFAR-10 图像
      - fake: 随机噪声图像（简单负样本）
    目标：让真实图像能量更低，噪声图像能量更高。
    """
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = select_device()
    else:
        device = torch.device(device)

    model = model.to(device)
    model.train()

    train_loader = get_data_loader(train=True, batch_size=batch_size)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"[Energy] Epoch {ep}/{epochs}")
        for real_images, _ in pbar:
            real_images = real_images.to(device)      # (B,3,64,64) 已在 [-1,1]
            B = real_images.size(0)
            total_samples += B

            # 负样本：纯噪声图像
            fake_images = torch.randn_like(real_images).clamp(-1.0, 1.0)

            # 正样本能量 & 负样本能量
            E_pos = model(real_images)   # (B,)
            E_neg = model(fake_images)   # (B,)

            loss = E_pos.mean() - E_neg.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            avg_loss = total_loss / max(1, total_samples)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        print(f"[Energy] Epoch {ep}/{epochs}  loss={avg_loss:.4f}")
        ckpt_path = os.path.join(out_dir, f"energy_ep{ep:03d}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"[Energy] Saved checkpoint to {ckpt_path}")

    print(f"[Energy] Training complete. Last weights in {ckpt_path}")
