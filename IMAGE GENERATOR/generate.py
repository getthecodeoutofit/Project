import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pathlib import Path
import logging
from typing import Optional
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class to store generation parameters"""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_dim = 100
        self.img_size = 28
        self.channels = 1
        self.checkpoint_path = Path("checkpoints/latest_checkpoint.pth")
        self.output_dir = Path("generated_images")
        self.batch_size = 16
        self.output_dir.mkdir(exist_ok=True)

class Generator(nn.Module):
    """Generator architecture matching the training implementation"""
    def __init__(self, config: Config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.noise_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, config.channels * config.img_size * config.img_size),
            nn.Tanh()
        )
        self.config = config

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.model(z)
        img = img.view(img.size(0), self.config.channels, self.config.img_size, self.config.img_size)
        return img

class ImageGenerator:
    """Class to handle image generation and visualization"""
    def __init__(self, config: Config):
        self.config = config
        self.device = config.device
        self.generator = self._load_generator()
        logger.info(f"Generator loaded successfully on {self.device}")

    def _load_generator(self) -> Generator:
        """Load the generator model from checkpoint"""
        try:
            generator = Generator(self.config).to(self.device)
            if not self.config.checkpoint_path.exists():
                raise FileNotFoundError(f"No checkpoint found at {self.config.checkpoint_path}")
            
            checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)
            generator.load_state_dict(checkpoint['generator'])
            generator.eval()
            return generator
        except Exception as e:
            logger.error(f"Failed to load generator: {e}")
            raise

    def generate_images(self, num_images: Optional[int] = None) -> torch.Tensor:
        """Generate images using the loaded generator"""
        try:
            num_images = num_images or self.config.batch_size
            with torch.no_grad():
                noise = torch.randn(num_images, self.config.noise_dim, device=self.device)
                generated_images = self.generator(noise)
            return generated_images
        except Exception as e:
            logger.error(f"Error generating images: {e}")
            raise

    def save_grid(self, images: torch.Tensor, filename: Optional[str] = None) -> None:
        """Save generated images as a grid"""
        try:
            # Rescale images to [0, 1] for visualization
            images = (images + 1) / 2
            grid = make_grid(images, nrow=4, normalize=False)
            grid_img = np.transpose(grid.cpu(), (1, 2, 0))

            # Create filename with timestamp if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_grid_{timestamp}.png"
            
            filepath = self.config.output_dir / filename
            
            plt.figure(figsize=(10, 10))
            plt.imshow(grid_img, cmap='gray')
            plt.axis('off')
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            logger.info(f"Saved image grid to {filepath}")
        except Exception as e:
            logger.error(f"Error saving image grid: {e}")
            raise

def setup_environment() -> None:
    """Setup the environment with proper error handling"""
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("CUDA is available. Using GPU.")
        else:
            logger.info("CUDA is not available. Using CPU.")
            
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
            
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        raise

def main() -> None:
    """Main function to generate images"""
    try:
        # Setup environment
        setup_environment()
        
        # Initialize configuration
        config = Config()
        
        # Create generator instance
        generator = ImageGenerator(config)
        
        # Generate and save images
        logger.info("Generating images...")
        images = generator.generate_images()
        generator.save_grid(images)
        logger.info("Image generation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
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