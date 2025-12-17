import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class to store all parameters"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 20
        self.batch_size = 128
        self.lr = 0.0002
        self.noise_dim = 100
        self.img_size = 28
        self.channels = 1
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_freq = 1

class Generator(nn.Module):
    """DCGAN Generator"""
    def __init__(self, config: Config):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(config.noise_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, config.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """DCGAN Discriminator"""
    def __init__(self, config: Config):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(config.channels, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class GANTrainer:
    """GAN Trainer for MNIST"""
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.setup_data()
        self.setup_models()
        self.setup_training()
        
    def setup_data(self) -> None:
        """Set up data pipeline with augmentation"""
        transform = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        try:
            dataset = datasets.MNIST(root='./data', download=True, transform=transform)
            self.dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True
            )
        except Exception as e:
            logger.error(f"Error setting up data: {e}")
            raise

    def setup_models(self) -> None:
        """Initialize models and load checkpoints if available"""
        self.generator = Generator(self.config).to(self.device)
        self.discriminator = Discriminator(self.config).to(self.device)
        
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load checkpoint with proper error handling"""
        checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pth"
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.generator.load_state_dict(checkpoint['generator'])
                self.discriminator.load_state_dict(checkpoint['discriminator'])
                logger.info("Loaded checkpoint from: %s", checkpoint_path)
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                # Not raising error to allow training from scratch

    def setup_training(self) -> None:
        """Initialize training components"""
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.config.lr,
            betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.lr,
            betas=(0.5, 0.999)
        )

    def save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint"""
        if epoch % self.config.checkpoint_freq == 0:
            try:
                checkpoint = {
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'epoch': epoch
                }
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                torch.save(checkpoint, checkpoint_path)
                torch.save(checkpoint, self.checkpoint_dir / "latest_checkpoint.pth")
                logger.info(f"Saved checkpoint at epoch {epoch}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
                raise

    def train(self) -> None:
        """Main training loop"""
        try:
            for epoch in range(self.config.num_epochs):
                self._train_epoch(epoch)
                self.save_checkpoint(epoch)
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.save_checkpoint(epoch)
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

    def _train_epoch(self, epoch: int) -> None:
        """Train a single epoch"""
        for i, (real_imgs, _) in enumerate(self.dataloader):
            try:
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                
                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                d_loss = self.train_discriminator(real_imgs, real_labels, fake_labels)
                
                # Train Generator
                self.optimizer_G.zero_grad()
                g_loss = self.train_generator(batch_size, real_labels)
                
                if i % 100 == 0:
                    logger.info(
                        f"Epoch [{epoch}/{self.config.num_epochs}], "
                        f"Step [{i}/{len(self.dataloader)}], "
                        f"D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}"
                    )
            except Exception as e:
                logger.error(f"Error in training step: {e}")
                continue

    def train_discriminator(self, real_imgs: torch.Tensor, 
                          real_labels: torch.Tensor, 
                          fake_labels: torch.Tensor) -> float:
        """Train discriminator"""
        # Real images
        real_output = self.discriminator(real_imgs)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(len(real_imgs), self.config.noise_dim, 1, 1, device=self.device)
        with torch.no_grad():
            fake_imgs = self.generator(noise)
        fake_output = self.discriminator(fake_imgs)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item()

    def train_generator(self, batch_size: int, real_labels: torch.Tensor) -> float:
        """Train generator"""
        noise = torch.randn(batch_size, self.config.noise_dim, 1, 1, device=self.device)
        fake_imgs = self.generator(noise)
        
        fake_output = self.discriminator(fake_imgs)
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item()

def setup_environment() -> None:
    """Setup the environment"""
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("CUDA is available. Using GPU.")
        else:
            logger.info("CUDA is not available. Using CPU.")
            
        Path("checkpoints").mkdir(exist_ok=True)
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise

def main() -> None:
    """Main function"""
    try:
        setup_environment()
        
        config = Config()
        
        trainer = GANTrainer(config)
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    main()