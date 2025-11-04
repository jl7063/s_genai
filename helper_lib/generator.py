# helper_lib/generator.py
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def generate_samples(model, device=None, num_samples=16):
    model.eval()
    if device is None:
        device = torch.device("mps") if torch.backends.mps.is_available() \
            else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    z = torch.randn(num_samples, model.z_dim, device=device)
    with torch.no_grad():
        imgs = model.generator(z).cpu()  # [N,1,28,28], Tanh 输出在 [-1,1]
        imgs = (imgs + 1) / 2.0  # 还原到 [0,1]

    grid = make_grid(imgs, nrow=int(num_samples ** 0.5), normalize=False)
    plt.figure(figsize=(6,6))
    plt.axis("off")
    plt.title("GAN Samples (MNIST)")
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.show()


