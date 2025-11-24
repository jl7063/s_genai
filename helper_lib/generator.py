# helper_lib/generator.py
from typing import Optional

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from .trainer import select_device, diffusion_schedule


# =========================================================
# 1) GAN (MNIST) sample generation — 保留你原来的功能
# =========================================================
def generate_gan_samples(
    model,
    device: Optional[torch.device] = None,
    num_samples: int = 16,
    show: bool = True,
):
    """
    使用已经训练好的 MNIST GAN 生成样本并显示。
    model: 包含 .generator 和 .z_dim 的 GAN 包装器 (helper_lib/model.py 里的 GAN 类)
    """
    if device is None:
        device = select_device()

    model.eval()
    model.generator.to(device)

    z = torch.randn(num_samples, model.z_dim, device=device)
    with torch.no_grad():
        imgs = model.generator(z).cpu()  # [N,1,28,28], Tanh 输出在 [-1,1]
        imgs = (imgs + 1) / 2.0  # 还原到 [0,1]

    if show:
        grid = make_grid(imgs, nrow=int(num_samples ** 0.5), normalize=False)
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title("GAN Samples (MNIST)")
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.show()

    return imgs  # 返回 Tensor，方便需要时用


# 兼容你原来的函数名 generate_samples
def generate_samples(
    model,
    device: Optional[torch.device] = None,
    num_samples: int = 16,
    show: bool = True,
):
    return generate_gan_samples(model, device=device, num_samples=num_samples, show=show)


# =========================================================
# 2) Diffusion UNet sample generation
#    从纯噪声开始，用多步反向扩散逐渐生成图片
# =========================================================
@torch.no_grad()
def generate_diffusion_samples(
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    num_samples: int = 16,
    diffusion_steps: int = 100,
    image_size: int = 64,
    num_channels: int = 3,
    show: bool = True,
):
    """
    使用训练好的 Diffusion UNet 从噪声中生成 CIFAR10 风格图片。

    假设:
    - model 是 helper_lib/model.py 里的 UNet( image_size=?, num_channels=3 )
    - 训练时图像已经归一化到 [-1,1]
    - 我们使用与 train_diffusion 相同的 diffusion_schedule

    返回:
      imgs: (num_samples, 3, H, W) in [0,1]
    """
    if device is None:
        device = select_device()

    model = model.to(device)
    model.eval()

    # 1) 从标准正态分布采样初始噪声图像
    x = torch.randn(num_samples, num_channels, image_size, image_size, device=device)

    # 2) 反向扩散：从 t=1 -> 0，离散成 diffusion_steps 个时间点
    for step in reversed(range(1, diffusion_steps + 1)):
        # 当前时间 t_s 和前一个时间 t_{s-1}
        t = torch.full((num_samples, 1, 1, 1), fill_value=step / diffusion_steps, device=device)
        if step > 1:
            t_prev = torch.full((num_samples, 1, 1, 1), fill_value=(step - 1) / diffusion_steps, device=device)
        else:
            t_prev = torch.zeros_like(t)

        # 当前时间的 noise_rate / signal_rate
        noise_rates, signal_rates = diffusion_schedule(t)
        # 前一个时间点的 noise_rate / signal_rate（用于重构 x_{t-1}）
        noise_rates_prev, signal_rates_prev = diffusion_schedule(t_prev)

        # UNet 预测当前噪声 eps_hat(x_t, t)
        pred_noises = model(x, noise_rates)

        # 估计干净图像 x0_hat
        # x_t = signal * x0 + noise * eps
        # => x0_hat = (x_t - noise * eps_hat) / signal
        x0_hat = (x - noise_rates * pred_noises) / (signal_rates + 1e-7)

        # 如果不是最后一步，可以重新加入一点噪声，走到上一个时间点
        if step > 1:
            # 重新构造下一个 x：x_{t_prev} = signal_prev * x0_hat + noise_prev * eps'
            eps_prime = torch.randn_like(x)
            x = signal_rates_prev * x0_hat + noise_rates_prev * eps_prime
        else:
            # 最后一步直接用去噪后的 x0_hat
            x = x0_hat

    # 3) 把 [-1,1] 映射回 [0,1] 以便显示
    x = x.clamp(-1, 1)
    imgs = (x + 1) / 2.0

    if show:
        grid = make_grid(imgs.cpu(), nrow=int(num_samples ** 0.5), normalize=False)
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title("Diffusion Samples")
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

    return imgs


# =========================================================
# 3) Energy-Based Model sample generation
#    对输入图片做“梯度下降”以降低 energy
# =========================================================
def generate_energy_samples(
    energy_model: torch.nn.Module,
    device: Optional[torch.device] = None,
    num_samples: int = 16,
    steps: int = 60,
    step_size: float = 10.0,
    noise_std: float = 0.005,
    image_size: int = 64,
    num_channels: int = 3,
    show: bool = True,
):
    """
    使用训练好的 EBM (EnergyModel) 通过对输入图片做梯度下降 (Langevin dynamics) 生成样本。

    思路：
      1. 初始化 x ~ N(0, I) （或 Uniform[-1,1]）
      2. 多次循环：
         - 给 x 加一点高斯噪声（探索）
         - 计算 energy E(x)
         - 对 x 求梯度 ∂E/∂x
         - x ← x - step_size * grad，并裁剪到 [-1,1]
      3. 把结果映射到 [0,1]，显示或返回

    注意:
      - 我们只对输入 x 求导，模型参数不更新。
    """
    if device is None:
        device = select_device()

    energy_model = energy_model.to(device)
    energy_model.eval()

    # 冻结模型参数，只对 x 求梯度
    for p in energy_model.parameters():
        p.requires_grad_(False)

    # 1) 初始化 x 为高斯噪声，并确保在 [-1,1] 区间
    x = torch.randn(num_samples, num_channels, image_size, image_size, device=device)
    x = x.clamp(-1.0, 1.0)

    # 确保后续可以对 x 求导
    x.requires_grad_(True)

    for _ in range(steps):
        # 每一步先加一点高斯噪声，类似 Langevin dynamics
        with torch.no_grad():
            noise = noise_std * torch.randn_like(x)
            x.add_(noise)
            x.clamp_(-1.0, 1.0)

        # 清空上一轮的梯度
        if x.grad is not None:
            x.grad.zero_()

        # 2) 计算 energy 并对 x 求梯度
        energy = energy_model(x)  # (B,) 或 (B,1)
        if energy.ndim > 1:
            energy = energy.view(energy.size(0), -1).mean(dim=1)
        total_energy = energy.sum()

        total_energy.backward()

        with torch.no_grad():
            # 梯度下降：沿着降低 energy 的方向更新 x
            grad = x.grad
            grad = grad.clamp(-0.03, 0.03)  # 限制步长，防止发散
            x.add_(-step_size * grad)
            x.clamp_(-1.0, 1.0)

    # 3) 映射到 [0,1]
    x = x.detach().clamp(-1, 1)
    imgs = (x + 1) / 2.0

    if show:
        grid = make_grid(imgs.cpu(), nrow=int(num_samples ** 0.5), normalize=False)
        plt.figure(figsize=(6, 6))
        plt.axis("off")
        plt.title("Energy-Based Model Samples")
        plt.imshow(grid.permute(1, 2, 0))
        plt.show()

    return imgs
