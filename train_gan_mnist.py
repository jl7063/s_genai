# train_gan_mnist.py
import torch
from helper_lib.model import get_model
from helper_lib.trainer import train_gan, select_device, generate_gan_samples

def main():
    device = select_device()
    print("Device:", device)

    # Build GAN model (MNIST)
    model = get_model("gan", z_dim=100)

    # Train GAN (trainer will build its own MNIST dataloader)
    train_gan(
        model,
        epochs=5,          # you can increase later, e.g., 20
        lr=2e-4,
        beta1=0.5,
        batch_size=128,    # reduce if VRAM/MPS memory errors
        out_dir="./artifacts_gan",
        device=None,       # keep None to auto-pick MPS/CUDA/CPU
        num_workers=2,
        sample_every=1,    # save grid each epoch
    )

    # Optionally, save one more sample grid after training
    generate_gan_samples(
        model,
        num_samples=16,
        out_dir="./artifacts_gan",
        filename="final_grid.png",
        device=device,
    )

if __name__ == "__main__":
    main()
