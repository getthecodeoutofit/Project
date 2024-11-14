import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Parameters
noise_dim = 100
img_size = 64
channels = 3   # RGB images have 3 channels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.conv_block(x)

# Define Generator
class Generator(nn.Module):
    def __init__(self, noise_dim):
        super(Generator, self).__init__()
        self.init_size = img_size // 4  # Initial size after initial convolution
        self.fc = nn.Linear(noise_dim, 128 * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            ResidualBlock(128, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

# Load generator model
generator = Generator(noise_dim).to(device)
generator.load_state_dict(torch.load('generator_model.pth', map_location=device))
generator.eval()  # Set to evaluation mode

# Generate and visualize images
def generate_images(num_images=16):
    noise = torch.randn(num_images, noise_dim, device=device)
    with torch.no_grad():  # Disable gradients for inference
        fake_images = generator(noise)
    fake_images = (fake_images + 1) / 2  # Rescale images to [0, 1] for visualization
    grid_img = np.transpose(make_grid(fake_images, nrow=4).cpu(), (1, 2, 0))

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_img)
    plt.axis("off")
    plt.show()

# Generate and display images
generate_images()
