import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import os
from datetime import datetime
import sys
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class to store monitoring parameters"""
    def __init__(self):
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", "1212"),
            "database": os.getenv("DB_NAME", "IMAGE")
        }
        self.output_dir = Path("performance_plots")
        self.output_dir.mkdir(exist_ok=True)
        self.plot_style = 'darkgrid'
        self.figure_size = (15, 8)
        self.save_format = 'png'
        self.dpi = 300
        self.moving_average_window = 50

class Database:
    """Database handler class with context manager support"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = None
        
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
            connection_string = (
                f"mysql+mysqlconnector://{self.config['user']}:{self.config['password']}"
                f"@{self.config['host']}/{self.config['database']}"
            )
            self.engine = create_engine(connection_string)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to the database")
        except Exception as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def fetch_metrics(self) -> pd.DataFrame:
        """Fetch training metrics from database with error handling"""
        try:
            query = """
                SELECT epoch, step, d_loss, g_loss, timestamp
                FROM TrainingMetrics
                ORDER BY epoch ASC, step ASC
            """
            df = pd.read_sql(text(query), con=self.engine)
            logger.info(f"Successfully fetched {len(df)} records from database")
            return df
        except Exception as e:
            logger.error(f"Error fetching metrics: {e}")
            raise

    def close(self) -> None:
        """Safely close database connection"""
        if self.engine is not None:
            self.engine.dispose()
            logger.info("Database connection closed")

class PerformanceAnalyzer:
    """Class to handle performance analysis and visualization"""
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.db_config)
        sns.set_style(config.plot_style)
        
    def calculate_moving_average(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving averages for loss values"""
        window = self.config.moving_average_window
        df['d_loss_ma'] = df['d_loss'].rolling(window=window, min_periods=1).mean()
        df['g_loss_ma'] = df['g_loss'].rolling(window=window, min_periods=1).mean()
        return df

    def plot_training_progress(self, df: pd.DataFrame, save: bool = True) -> None:
        """Plot training progress with improved visualization"""
        try:
            df = self.calculate_moving_average(df)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size)
            
            # Plot Discriminator Loss
            sns.scatterplot(data=df, x='step', y='d_loss', alpha=0.1, color='blue', ax=ax1)
            sns.lineplot(data=df, x='step', y='d_loss_ma', color='darkblue', ax=ax1)
            ax1.set_title('Discriminator Loss over Training Steps')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Loss')
            
            # Plot Generator Loss
            sns.scatterplot(data=df, x='step', y='g_loss', alpha=0.1, color='red', ax=ax2)
            sns.lineplot(data=df, x='step', y='g_loss_ma', color='darkred', ax=ax2)
            ax2.set_title('Generator Loss over Training Steps')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            
            plt.tight_layout()
            
            if save:
                self._save_plot(fig, 'training_progress')
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting training progress: {e}")
            raise
        finally:
            plt.close()

    def plot_loss_distributions(self, df: pd.DataFrame, save: bool = True) -> None:
        """Plot loss value distributions"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size)
            
            # Discriminator Loss Distribution
            sns.histplot(data=df, x='d_loss', bins=50, ax=ax1)
            ax1.set_title('Discriminator Loss Distribution')
            ax1.set_xlabel('Loss Value')
            
            # Generator Loss Distribution
            sns.histplot(data=df, x='g_loss', bins=50, ax=ax2)
            ax2.set_title('Generator Loss Distribution')
            ax2.set_xlabel('Loss Value')
            
            plt.tight_layout()
            
            if save:
                self._save_plot(fig, 'loss_distributions')
            else:
                plt.show()
                
        except Exception as e:
            logger.error(f"Error plotting loss distributions: {e}")
            raise
        finally:
            plt.close()

    def generate_performance_report(self, df: pd.DataFrame) -> None:
        """Generate a comprehensive performance report"""
        try:
            # Calculate key metrics
            metrics = {
                'Total Training Steps': len(df),
                'Number of Epochs': df['epoch'].max() + 1,
                'Average D Loss': df['d_loss'].mean(),
                'Average G Loss': df['g_loss'].mean(),
                'D Loss Std Dev': df['d_loss'].std(),
                'G Loss Std Dev': df['g_loss'].std(),
                'Training Duration (hours)': (
                    df['timestamp'].max() - df['timestamp'].min()
                ).total_seconds() / 3600
            }
            
            # Save metrics to file
            report_path = self.config.output_dir / 'performance_report.txt'
            with open(report_path, 'w') as f:
                f.write("GAN Training Performance Report\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                for metric, value in metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
                    
            logger.info(f"Performance report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            raise

    def _save_plot(self, fig: plt.Figure, name: str) -> None:
        """Save plot with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.{self.config.save_format}"
        filepath = self.config.output_dir / filename
        fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {filepath}")

def main() -> None:
    """Main function to analyze and visualize training performance"""
    try:
        # Initialize configuration
        config = Config()
        
        # Create analyzer instance
        analyzer = PerformanceAnalyzer(config)
        
        # Fetch metrics and generate visualizations
        with analyzer.db.connection_scope():
            logger.info("Fetching training metrics...")
            df = analyzer.db.fetch_metrics()
            
            logger.info("Generating performance visualizations...")
            analyzer.plot_training_progress(df)
            analyzer.plot_loss_distributions(df)
            analyzer.generate_performance_report(df)
            
        logger.info("Performance analysis completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()