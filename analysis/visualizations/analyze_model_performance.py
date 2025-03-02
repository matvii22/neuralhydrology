"""
Analyze and visualize the performance of trained neural hydrology models.

This script provides functionality to:
1. Evaluate model performance on test data
2. Generate performance visualizations
3. Calculate and summarize metrics across basins
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.evaluation import metrics
from neuralhydrology.utils.config import Config
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HydrologyModelAnalyzer:
    """Class to analyze and visualize neural hydrology model performance."""
    
    def __init__(self, run_dir: str, epoch: int = 5):
        """
        Initialize the analyzer with run directory and epoch.
        
        Args:
            run_dir: Path to the model run directory
            epoch: Which model epoch to analyze
        """
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise ValueError(f"Run directory {run_dir} does not exist")
            
        config_file = self.run_dir / "config.yml"
        if not config_file.exists():
            raise ValueError(f"Config file not found at {config_file}")
            
        self.epoch = epoch
        self.results = None
        self.cfg = Config(config_file)
        
        # Create output directories
        self.output_dir = self.run_dir / "analysis"
        self.plot_dir = self.output_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for consistent visualizations
        try:
            plt.style.use('default')  # Reset to default style
            sns.set_theme(style="whitegrid", font_scale=1.1)
            sns.set_palette("deep")
        except Exception as e:
            logger.warning(f"Could not set plot style: {e}. Using default style.")
        
    def load_results(self, period: str = "test") -> None:
        """
        Load model evaluation results from pickle file.
        
        Args:
            period: Which period to analyze ('test', 'train', or 'validation')
        """
        if period not in ["test", "train", "validation"]:
            raise ValueError(f"Invalid period {period}. Must be one of: test, train, validation")
            
        # Run evaluation if results don't exist
        results_file = self.run_dir / period / f"model_epoch{self.epoch:03d}" / f"{period}_results.p"
        if not results_file.exists():
            logger.info(f"Running evaluation for {period} period...")
            try:
                start_evaluation(cfg=self.cfg, run_dir=self.run_dir, epoch=self.epoch, period=period)
            except Exception as e:
                raise RuntimeError(f"Failed to run evaluation: {e}")
            
        # Load results
        logger.info(f"Loading results from {results_file}")
        try:
            with open(results_file, "rb") as fp:
                self.results = pickle.load(fp)
        except Exception as e:
            raise RuntimeError(f"Failed to load results from {results_file}: {e}")
            
        if not self.results:
            raise ValueError("No results found after evaluation")
            
        logger.info(f"Loaded results for {len(self.results)} basins")
    
    def get_basin_metrics(self, basin: str) -> Dict[str, float]:
        """Get all available metrics for a specific basin."""
        if basin not in self.results:
            raise ValueError(f"Basin {basin} not found in results")
            
        metrics = {}
        for metric_name in ['NSE', 'KGE', 'Alpha-NSE', 'Beta-NSE']:
            try:
                metrics[metric_name] = float(self.results[basin]['1D'][metric_name])
            except (KeyError, TypeError):
                metrics[metric_name] = np.nan
        return metrics
    
    def plot_basin_hydrograph(self, basin: str, save: bool = True) -> None:
        """
        Create hydrograph plot for a specific basin.
        
        Args:
            basin: Basin ID to plot
            save: Whether to save the plot to disk
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")
            
        if basin not in self.results:
            raise ValueError(f"Basin {basin} not found in results")
            
        try:
            # Extract data
            xr_data = self.results[basin]['1D']['xr']
            qobs = xr_data['discharge_spec_obs']
            qsim = xr_data['discharge_spec_sim']
            
            # Get metrics
            metrics_dict = self.get_basin_metrics(basin)
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in metrics_dict.items() if not np.isnan(v)])
            
            # Create plot
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(qobs['date'], qobs, label='Observed', alpha=0.8, linewidth=1.5)
            ax.plot(qsim['date'], qsim, label='Simulated', alpha=0.8, linewidth=1.5)
            
            # Style the plot
            ax.set_ylabel("Specific Discharge", fontsize=12)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_title(f"Basin {basin}\n{metrics_str}", fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if save:
                plt.savefig(self.plot_dir / f"{basin}_hydrograph.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Failed to plot hydrograph for basin {basin}: {e}")
            plt.close()  # Ensure figure is closed even if plotting fails
    
    def plot_metric_distribution(self, metric: str = 'NSE', save: bool = True) -> None:
        """
        Plot distribution of performance metrics across all basins.
        
        Args:
            metric: Which metric to plot
            save: Whether to save the plot to disk
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")
            
        try:
            # Collect metric values
            values = []
            for basin in self.results.keys():
                try:
                    val = float(self.results[basin]['1D'][metric])
                    if not np.isnan(val):
                        values.append(val)
                except (KeyError, TypeError):
                    continue
            
            if not values:
                logger.warning(f"No valid values found for metric {metric}")
                return
                
            # Create distribution plot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(values, kde=True, ax=ax)
            
            # Style the plot
            ax.set_xlabel(metric, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_title(f"Distribution of {metric} across basins", fontsize=14)
            
            # Add summary statistics
            stats = f"Mean: {np.mean(values):.3f}\nMedian: {np.median(values):.3f}\nN: {len(values)}"
            ax.text(0.95, 0.95, stats, transform=ax.transAxes, 
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(facecolor='white', alpha=0.8))
            
            if save:
                plt.savefig(self.plot_dir / f"{metric}_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
        except Exception as e:
            logger.error(f"Failed to plot metric distribution for {metric}: {e}")
            plt.close()  # Ensure figure is closed even if plotting fails
            
    def summarize_metrics(self) -> pd.DataFrame:
        """
        Create summary DataFrame of all metrics across basins.
        
        Returns:
            DataFrame with metric summaries
        """
        if self.results is None:
            raise ValueError("No results loaded. Call load_results() first.")
            
        try:
            # Calculate metrics for each basin
            metrics_dict = {}
            for basin in self.results.keys():
                basin_metrics = self.get_basin_metrics(basin)
                if any(not np.isnan(v) for v in basin_metrics.values()):
                    metrics_dict[basin] = basin_metrics
            
            if not metrics_dict:
                raise ValueError("No valid metrics found for any basin")
                
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(metrics_dict, orient='index')
            
            # Calculate summary statistics
            summary = pd.DataFrame({
                'mean': df.mean(),
                'median': df.median(),
                'std': df.std(),
                'min': df.min(),
                'max': df.max(),
                'count': df.count()  # Number of non-NaN values
            })
            
            # Save detailed metrics
            df.to_csv(self.output_dir / "basin_metrics.csv")
            
            return summary
        except Exception as e:
            raise RuntimeError(f"Failed to calculate metric summary: {e}")

def main():
    """Main execution function with command line arguments."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze neural hydrology model performance')
    parser.add_argument('--run_dir', type=str, required=True,
                      help='Path to the model run directory')
    parser.add_argument('--epoch', type=int, required=True,
                      help='Which model epoch to analyze')
    args = parser.parse_args()

    try:
        # Initialize analyzer with command line arguments
        analyzer = HydrologyModelAnalyzer(args.run_dir, args.epoch)
        
        # Load results
        analyzer.load_results()
        
        # Generate plots for each basin
        for basin in analyzer.results.keys():
            analyzer.plot_basin_hydrograph(basin)
            
        # Plot metric distributions
        for metric in ['NSE', 'KGE', 'Alpha-NSE', 'Beta-NSE']:
            analyzer.plot_metric_distribution(metric)
            
        # Generate and save summary
        summary = analyzer.summarize_metrics()
        summary.to_csv(analyzer.output_dir / "metric_summary.csv")
        
        logger.info(f"Analysis complete. Results saved to {analyzer.output_dir}")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main() 