import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 100
num_channels = 1
image_size = 28 * 28

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
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x).view(-1, num_channels, 28, 28)

generator = Generator().to(device)
generator.load_state_dict(torch.load("gan_mnist_generator.pth"))
generator.eval()

os.makedirs("gan_mnist_output", exist_ok=True)
with torch.no_grad():
    noise = torch.randn(16, latent_dim).to(device)
    generated_images = generator(noise).cpu()
    save_image(generated_images, "gan_mnist_output/generated_samples.png", normalize=True, nrow=4)

print("Generated samples saved to gan_mnist_output/generated_samples.png")
