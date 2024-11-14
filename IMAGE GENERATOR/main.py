import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import mysql.connector

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Parameters
num_epochs = 100
batch_size = 64
lr = 0.0002
noise_dim = 100
img_size = 64
channels = 3  # RGB images have 3 channels

# Load CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB images to [-1, 1]
])

dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1212",
    database="project"
)
cursor = db.cursor()

# Create table for training metrics
cursor.execute("""
    CREATE TABLE IF NOT EXISTS TrainingMetrics (
        epoch INT,
        step INT,
        d_loss FLOAT,
        g_loss FLOAT,
        PRIMARY KEY (epoch, step)
    )
""")

def log_metrics_to_db(epoch, step, d_loss, g_loss):
    cursor.execute("INSERT INTO TrainingMetrics (epoch, step, d_loss, g_loss) VALUES (%s, %s, %s, %s)", 
                   (epoch, step, d_loss, g_loss))
    db.commit()

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

# Define Generator with Residual Blocks
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

# Define Discriminator with Spectral Normalization
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(-1, 1).squeeze(1)

# Instantiate models
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training Loop
for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Prepare real and fake labels
        real_imgs = imgs.to(device)
        real_labels = torch.ones(imgs.size(0), device=device)
        fake_labels = torch.zeros(imgs.size(0), device=device)

        # Train Discriminator
        optimizer_D.zero_grad()
        outputs = discriminator(real_imgs)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()

        noise = torch.randn(imgs.size(0), noise_dim, device=device)
        fake_imgs = generator(noise)
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        # Log metrics to database and print progress
        log_metrics_to_db(epoch, i, d_loss_real.item() + d_loss_fake.item(), g_loss.item())
        
        if i % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                  f"D Loss: {d_loss_real + d_loss_fake:.4f}, G Loss: {g_loss:.4f}")

# Save generator model
torch.save(generator.state_dict(), 'generator_model.pth')
print("Model saved as 'generator_model.pth'.")
