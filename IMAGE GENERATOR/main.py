import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import logging
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from typing import Optional, Dict, Any
import sys
from contextlib import contextmanager
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
        self.num_epochs = 10
        self.batch_size = 16
        self.lr = 0.0002
        self.noise_dim = 100
        self.img_size = 256
        self.channels = 3
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_freq = 2
        # Move sensitive information to environment variables
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", "1212"),
            "database": os.getenv("DB_NAME", "IMAGE")
        }

class Database:
    """Database handler class with context manager support"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection: Optional[mysql.connector.MySQLConnection] = None
        self.cursor: Optional[mysql.connector.cursor.MySQLCursor] = None
    
    @contextmanager
    def connection_scope(self):
        """Context manager for database connections"""
        try:
            self.connect()
            yield
        finally:
            self.close()
            
    def connect(self) -> None:
        """Establish database connection with proper error handling"""
        try:
            self.connection = mysql.connector.connect(**self.config)
            if not self.connection.is_connected():
                raise Error("Failed to connect to database")
            self.cursor = self.connection.cursor()
            self._create_tables()
            logger.info("Successfully connected to the database")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def _create_tables(self) -> None:
        """Create necessary database tables if they don't exist"""
        try:
            # Drop the existing table if it exists
            self.cursor.execute("DROP TABLE IF EXISTS TrainingMetrics")
            
            # Create the table with a proper auto-incrementing primary key
            self.cursor.execute("""
                CREATE TABLE if not Exists TrainingMetrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    epoch INT,
                    step INT,
                    d_loss FLOAT,
                    g_loss FLOAT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_epoch_step (epoch, step)
                )
            """)
            self.connection.commit()
        except Error as e:
            logger.error(f"Error creating tables: {e}")
            raise

    def log_metrics(self, epoch: int, step: int, d_loss: float, g_loss: float) -> None:
        """Log training metrics to database with error handling and retry logic"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                if not self.cursor or not self.connection.is_connected():
                    self.connect()
                
                self.cursor.execute(
                    """INSERT INTO TrainingMetrics 
                       (epoch, step, d_loss, g_loss) 
                       VALUES (%s, %s, %s, %s)""",
                    (epoch, step, float(d_loss), float(g_loss))
                )
                self.connection.commit()
                break
            except Error as e:
                retry_count += 1
                logger.warning(f"Attempt {retry_count} failed to log metrics: {e}")
                if retry_count == max_retries:
                    logger.error("Failed to log metrics after maximum retries")
                    raise
                self.connect()  # Try to reconnect before next attempt

    def close(self) -> None:
        """Safely close database connections"""
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection and self.connection.is_connected():
                self.connection.close()
                logger.info("Database connection closed successfully")
        except Error as e:
            logger.error(f"Error closing database connection: {e}")

class ResidualBlock(nn.Module):
    """Improved Residual Block with proper initialization"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.skip = (nn.Identity() if in_channels == out_channels 
                    else nn.Conv2d(in_channels, out_channels, 1))
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.conv_block(x)

class Generator(nn.Module):
    """Improved Generator with proper initialization and input validation"""
    def __init__(self, config: Config):
        super().__init__()
        self.init_size = config.img_size // 4
        self.noise_dim = config.noise_dim
        
        self.fc = nn.Linear(config.noise_dim, 256 * self.init_size ** 2)
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.Upsample(scale_factor=2),
            ResidualBlock(256, 256),
            nn.Dropout2d(0.3),
            nn.Upsample(scale_factor=2),
            ResidualBlock(256, 128),
            nn.Dropout2d(0.3),
            nn.Conv2d(128, config.channels, 3, 1, 1),
            nn.Tanh()
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(1) != self.noise_dim:
            raise ValueError(f"Expected noise dimension {self.noise_dim}, got {z.size(1)}")
        out = self.fc(z)
        out = out.view(-1, 256, self.init_size, self.init_size)
        return self.conv_blocks(out)

class Discriminator(nn.Module):
    """Improved Discriminator with input validation and proper initialization"""
    def __init__(self, config: Config):
        super().__init__()
        self.expected_input_size = (config.channels, config.img_size, config.img_size)
        self.final_size = config.img_size // 16
        
        self.features = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(config.channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            nn.utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * self.final_size * self.final_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size()[1:] != self.expected_input_size:
            raise ValueError(f"Expected input size {self.expected_input_size}, got {x.size()[1:]}")
        x = self.features(x)
        return self.classifier(x)

class GANTrainer:
    """Improved GAN Trainer with better error handling and training stability"""
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.db = Database(config.db_config)
        self.setup_data()
        self.setup_models()
        self.setup_training()
        
    def setup_data(self) -> None:
        """Set up data pipeline with augmentation"""
        transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        try:
            dataset = datasets.CIFAR10(root='./data', download=True, transform=transform)
            self.dataloader = DataLoader(
                dataset, 
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
                drop_last=True  # Ensure consistent batch sizes
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
                checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=True)
                self.generator.load_state_dict(checkpoint['generator'])
                self.discriminator.load_state_dict(checkpoint['discriminator'])
                logger.info("Loaded checkpoint from: %s", checkpoint_path)
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                raise

    def setup_training(self) -> None:
        """Initialize training components with improved optimizer settings"""
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
        """Save training checkpoint with error handling"""
        if epoch % self.config.checkpoint_freq == 0:
            try:
                checkpoint = {
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'epoch': epoch
                }
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
                torch.save(checkpoint, checkpoint_path)
                # Also save as latest checkpoint
                torch.save(checkpoint, self.checkpoint_dir / "latest_checkpoint.pth")
                logger.info(f"Saved checkpoint at epoch {epoch}")
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
                raise

    def train(self) -> None:
        """Main training loop with improved error handling and monitoring"""
        with self.db.connection_scope():
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
        """Train a single epoch with improved metric logging"""
        for i, (real_imgs, _) in enumerate(self.dataloader):
            try:
                real_imgs = real_imgs.to(self.device)
                batch_size = real_imgs.size(0)
                
                # Create labels with noise for label smoothing
                real_labels = torch.ones(batch_size, 1, device=self.device) * 0.9
                fake_labels = torch.zeros(batch_size, 1, device=self.device)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                d_loss = self.train_discriminator(real_imgs, real_labels, fake_labels)
                
                # Train Generator
                self.optimizer_G.zero_grad()
                g_loss = self.train_generator(batch_size, real_labels)
                
                # Log metrics with error handling
                try:
                    self.db.log_metrics(epoch, i, d_loss, g_loss)
                except Exception as e:
                    logger.error(f"Failed to log metrics: {e}")
                    # Continue training even if logging fails
                    
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
        """Train discriminator with improved stability"""
        # Real images
        real_output = self.discriminator(real_imgs)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Fake images
        noise = torch.randn(len(real_imgs), self.config.noise_dim, device=self.device)
        with torch.no_grad():  # Don't compute gradients for generator during discriminator training
            fake_imgs = self.generator(noise)
        fake_output = self.discriminator(fake_imgs)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # Combined loss and gradient
        d_loss = (d_loss_real + d_loss_fake) / 2  # Average the losses
        d_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        self.optimizer_D.step()
        
        return d_loss.item()

    def train_generator(self, batch_size: int, real_labels: torch.Tensor) -> float:
        """Train generator with improved loss calculation"""
        # Generate fake images
        noise = torch.randn(batch_size, self.config.noise_dim, device=self.device)
        fake_imgs = self.generator(noise)
        
        # Calculate generator loss
        fake_output = self.discriminator(fake_imgs)
        g_loss = self.criterion(fake_output, real_labels)
        g_loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        self.optimizer_G.step()
        
        return g_loss.item()

    def generate_samples(self, num_samples: int = 16) -> torch.Tensor:
        """Generate sample images from the current state of the generator"""
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(num_samples, self.config.noise_dim, device=self.device)
            samples = self.generator(noise)
        self.generator.train()
        return samples

def setup_environment() -> None:
    """Setup the environment with proper error handling"""
    try:
        # Set up PyTorch environment
        if torch.cuda.is_available():
            # Set cuda to deterministic mode for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("CUDA is available. Using GPU.")
        else:
            logger.info("CUDA is not available. Using CPU.")
            
        # Ensure checkpoint directory exists
        Path("checkpoints").mkdir(exist_ok=True)
        
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise

def main() -> None:
    """Main function with improved error handling and setup"""
    try:
        # Setup environment
        setup_environment()
        
        # Initialize configuration
        config = Config()
        
        # Create and train the GAN
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
        # Cleanup
        torch.cuda.empty_cache()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    main()