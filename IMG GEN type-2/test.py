import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from pathlib import Path
import logging
import mysql.connector
from mysql.connector import Error
import os
import time
import psutil
import tqdm
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class Config:
    """Configuration class to store all parameters."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = 50
        self.batch_size = 16
        self.lr = 0.0002
        self.noise_dim = 100
        self.img_size = 64
        self.channels = 3
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_freq = 5

        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", "1212"),
            "database": os.getenv("DB_NAME", "IMAGE"),
        }

        self.early_stopping_patience = 10
        self.lr_scheduler_patience = 3
        self.sample_image_dir = Path("samples")
        self.custom_dataset_path = None
        self.dynamic_batch = True
        self.multi_gpu = torch.cuda.device_count() > 1

    def adjust_batch_size(self):
        """Dynamically adjusts batch size based on available GPU memory."""
        if self.dynamic_batch and self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory = total_memory - torch.cuda.memory_reserved(0)
            self.batch_size = max(16, int((free_memory / total_memory) * 128))
            logger.info(f"Dynamic batch size set to: {self.batch_size}")


class Database:
    """Database handler class with context manager support."""

    def __init__(self, config):
        self.config = config
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish database connection."""
        try:
            self.connection = mysql.connector.connect(**self.config)
            self.cursor = self.connection.cursor()
            self._create_tables()
            logger.info("Database connected.")
        except Error as e:
            logger.error(f"Database connection error: {e}")

    def _create_tables(self):
        """Create database tables if they don't exist."""
        try:
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS TrainingMetrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    epoch INT,
                    step INT,
                    d_loss FLOAT,
                    g_loss FLOAT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Checkpoints (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    epoch INT,
                    checkpoint_path VARCHAR(255),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            self.connection.commit()
        except Error as e:
            logger.error(f"Error creating tables: {e}")

    def log_metrics(self, epoch, step, d_loss, g_loss):
        """Log metrics to the database."""
        try:
            self.cursor.execute(
                """
                INSERT INTO TrainingMetrics (epoch, step, d_loss, g_loss)
                VALUES (%s, %s, %s, %s)
            """,
                (epoch, step, d_loss, g_loss),
            )
            self.connection.commit()
        except Error as e:
            logger.error(f"Error logging metrics: {e}")

    def log_checkpoint(self, epoch, checkpoint_path):
        """Log checkpoint details to the database."""
        try:
            self.cursor.execute(
                """
                INSERT INTO Checkpoints (epoch, checkpoint_path)
                VALUES (%s, %s)
            """,
                (epoch, checkpoint_path),
            )
            self.connection.commit()
        except Error as e:
            logger.error(f"Error logging checkpoint: {e}")

    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, config):
        super().__init__()
        self.init_size = config.img_size // 8
        self.fc = nn.Linear(config.noise_dim, 256 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, config.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.fc(z)
        out = out.view(out.size(0), 256, self.init_size, self.init_size)
        return self.conv_blocks(out)


class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, config):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(config.channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * (config.img_size // 16) ** 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# The rest of the trainer class would be similar but with additional hooks for GPU monitoring, dynamic adjustments,
# and better error handling.
class GANTrainer:
    """Trainer class for GAN."""

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.generator = Generator(config).to(self.device)
        self.discriminator = Discriminator(config).to(self.device)

        if config.multi_gpu:
            # Use multiple GPUs if available
            self.generator = nn.DataParallel(self.generator)
            self.discriminator = nn.DataParallel(self.discriminator)

        # Loss function
        self.criterion = nn.BCELoss()

        # Optimizers for both networks
        self.optimizer_G = optim.Adam(
            self.generator.parameters(), lr=config.lr, betas=(0.5, 0.999)
        )
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999)
        )

        # Learning rate schedulers
        self.lr_scheduler_G = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_G, patience=config.lr_scheduler_patience, verbose=True
        )
        self.lr_scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_D, patience=config.lr_scheduler_patience, verbose=True
        )

        # Load dataset
        self.dataset = self._load_dataset()
        self.dataloader = DataLoader(
            self.dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
        )

        # Connect to database
        self.db = Database(config.db_config)
        self.db.connect()

        # Epoch tracking and checkpoints
        self.start_epoch = 0
        self._load_checkpoint_if_exists()

        # Prepare for generating samples
        self.config.sample_image_dir.mkdir(exist_ok=True)

    def _load_dataset(self):
        """Load CIFAR-10 or custom dataset."""
        transform = transforms.Compose(
            [
                transforms.Resize((self.config.img_size, self.config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * self.config.channels, [0.5] * self.config.channels),
            ]
        )

        if self.config.custom_dataset_path:
            if not os.path.exists(self.config.custom_dataset_path):
                raise ValueError(f"Custom dataset path {self.config.custom_dataset_path} not found.")
            logger.info(f"Using custom dataset from {self.config.custom_dataset_path}")
            return datasets.ImageFolder(self.config.custom_dataset_path, transform=transform)
        else:
            logger.info("Using CIFAR-10 dataset.")
            return datasets.CIFAR10(root="./data", download=True, transform=transform)

    def _load_checkpoint_if_exists(self):
        """Load the latest checkpoint if available."""
        checkpoint_files = sorted(
            self.config.checkpoint_dir.glob("checkpoint_epoch_*.pth"), key=os.path.getmtime
        )
        if checkpoint_files:
            checkpoint_path = checkpoint_files[-1]
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load models and optimizers
            self.generator.load_state_dict(checkpoint["generator"])
            self.discriminator.load_state_dict(checkpoint["discriminator"])
            self.optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            self.optimizer_D.load_state_dict(checkpoint["optimizer_D"])
            self.start_epoch = checkpoint["epoch"] + 1

    def _save_checkpoint(self, epoch):
        """Save a checkpoint at the given epoch."""
        checkpoint_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        self.config.checkpoint_dir.mkdir(exist_ok=True)

        torch.save(
            {
                "epoch": epoch,
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "optimizer_G": self.optimizer_G.state_dict(),
                "optimizer_D": self.optimizer_D.state_dict(),
            },
            checkpoint_path,
        )

        logger.info(f"Checkpoint saved: {checkpoint_path}")
        self.db.log_checkpoint(epoch, str(checkpoint_path))

    def _generate_samples(self, epoch):
        """Generate sample images for visual inspection."""
        z = torch.randn(16, self.config.noise_dim, device=self.device)
        gen_imgs = self.generator(z).detach().cpu()
        grid = utils.make_grid(gen_imgs, normalize=True)
        sample_path = self.config.sample_image_dir / f"epoch_{epoch}.png"
        utils.save_image(grid, sample_path)
        logger.info(f"Sample images saved to {sample_path}")

    def _monitor_gpu_usage(self):
        """Monitor and log GPU memory usage."""
        if self.device.type == "cuda":
            gpu_memory = torch.cuda.memory_allocated(0) / 1024 ** 2
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 2
            logger.info(f"GPU Memory Usage: {gpu_memory:.2f} MB / {gpu_total:.2f} MB")

    def train(self):
        """Training loop for GAN."""
        best_g_loss = float("inf")
        early_stop_counter = 0

        for epoch in range(self.start_epoch, self.config.num_epochs):
            start_time = time.time()
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0

            self.generator.train()
            self.discriminator.train()

            for i, (real_imgs, _) in enumerate(tqdm.tqdm(self.dataloader, desc=f"Epoch {epoch}")):
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)

                # Labels for real and fake images
                valid = torch.ones(batch_size, 1, device=self.device)
                fake = torch.zeros(batch_size, 1, device=self.device)

                # Train Generator
                self.optimizer_G.zero_grad()
                z = torch.randn(batch_size, self.config.noise_dim, device=self.device)
                gen_imgs = self.generator(z)
                g_loss = self.criterion(self.discriminator(gen_imgs), valid)
                g_loss.backward()
                self.optimizer_G.step()

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_loss = self.criterion(self.discriminator(real_imgs), valid)
                fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_D.step()

                epoch_d_loss += d_loss.item()
                epoch_g_loss += g_loss.item()

                # Log metrics every 10 batches
                if i % 10 == 0:
                    logger.info(
                        f"Epoch [{epoch}/{self.config.num_epochs}] Batch [{i}/{len(self.dataloader)}] "
                        f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}"
                    )
                    self.db.log_metrics(epoch, i, d_loss.item(), g_loss.item())

            # Average losses for the epoch
            epoch_d_loss /= len(self.dataloader)
            epoch_g_loss /= len(self.dataloader)

            # Adjust learning rate
            self.lr_scheduler_G.step(epoch_g_loss)
            self.lr_scheduler_D.step(epoch_d_loss)

            # Generate samples
            self._generate_samples(epoch)

            # Save checkpoint periodically
            if epoch % self.config.checkpoint_freq == 0:
                self._save_checkpoint(epoch)

            # Early stopping
            if epoch_g_loss < best_g_loss:
                best_g_loss = epoch_g_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.config.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

            # Log GPU usage and epoch time
            self._monitor_gpu_usage()
            epoch_time = time.time() - start_time
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

        self.db.close()
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Load configuration
    config = Config()
    config.db_config = {
        "host": "localhost",
        "user": "root",                # Replace with your MySQL username
        "password": "1212",            # Replace with your MySQL password
        "database": "IMAGE"            # Replace with your database name
    }

    # Ensure necessary directories exist
    config.checkpoint_dir.mkdir(exist_ok=True)
    config.sample_image_dir.mkdir(exist_ok=True)

    # Initialize GAN trainer
    trainer = GANTrainer(config)

    try:
        # Start training
        trainer.train()
    except Exception as e:
        logger.error(f"Training interrupted due to an error: {e}")
    finally:
        # Ensure the database connection is closed properly
        trainer.db.close()
