import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
num_channels = 1
gan_image_size = 28 * 28

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, gan_image_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x).view(-1, num_channels, 28, 28)

gan_generator = Generator().to(device)
gan_generator.load_state_dict(torch.load("models/gan_mnist_generator.pth"))
gan_generator.eval()

with torch.no_grad():
    noise = torch.randn(8, latent_dim).to(device)
    gan_images = gan_generator(noise).cpu()

os.makedirs("comparison_output", exist_ok=True)
save_image(gan_images, "comparison_output/gan_samples.png", normalize=True, nrow=4)

stylegan3_dir = "stylegan3_output"
stylegan3_images = []
if os.path.exists(stylegan3_dir):
    for file in os.listdir(stylegan3_dir):
        if file.endswith(".png"):
            img_path = os.path.join(stylegan3_dir, file)
            img = Image.open(img_path)
            img_np = np.array(img) / 255.0
            stylegan3_images.append(img_np)
            if len(stylegan3_images) >= 8:
                break

if stylegan3_images:
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        if i < len(stylegan3_images):
            ax.imshow(stylegan3_images[i])
            ax.axis('off')
    plt.savefig("comparison_output/stylegan3_samples.png")
    plt.close()

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i in range(8):
    row = i // 4
    col = i % 4
    if i < len(gan_images):
        img = gan_images[i].permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        axes[row, col].imshow(img, cmap='gray' if num_channels == 1 else None)
        axes[row, col].set_title(f"GAN Sample {i+1}")
    elif i - 4 < len(stylegan3_images):
        axes[row, col].imshow(stylegan3_images[i - 4])
        axes[row, col].set_title(f"StyleGAN3 Sample {i-3}")
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig("comparison_output/comparison.png")
plt.close()

print("Comparison images saved to comparison_output/")
print("GAN samples: comparison_output/gan_samples.png")
print("StyleGAN3 samples: comparison_output/stylegan3_samples.png")
print("Combined comparison: comparison_output/comparison.png")
