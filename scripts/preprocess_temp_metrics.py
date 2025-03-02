import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TemperatureMetricsProcessor:
    """Process temperature-related metrics for CAMELS-GB dataset."""
    
    def __init__(self, data_dir: str):
        """
        Initialize processor with data directory.
        
        Args:
            data_dir: Path to CAMELS-GB data directory
        """
        self.data_dir = Path(data_dir)
        
    def calculate_temp_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate temperature-based metrics using rolling windows and temporal variations.
        
        Args:
            df: DataFrame with original CAMELS-GB timeseries data
            
        Returns:
            DataFrame with additional temperature metrics
        """
        # Ensure we have temperature and precipitation columns
        if 'temperature' not in df.columns:
            raise ValueError("DataFrame must contain 'temperature' column")
            
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # 1. Temperature Range and Variability (7-day window)
        rolling_temp = df['temperature'].rolling(window=7, min_periods=1)
        df['temp_7d_range'] = rolling_temp.max() - rolling_temp.min()
        df['temp_7d_std'] = rolling_temp.std()
        
        # 2. Temperature Persistence and Changes
        # Calculate day-to-day temperature changes
        df['temp_change'] = df['temperature'].diff()
        
        # Mark significant temperature changes (>2°C in either direction)
        df['significant_temp_change'] = (abs(df['temp_change']) > 2).astype(int)
        
        # 3. Rolling Temperature Statistics
        df['temp_7d_mean'] = rolling_temp.mean()
        
        # 4. Snow-Related Metrics
        # a) Snow Probability (based on temperature and recent conditions)
        snow_conditions = (
            (df['temperature'] < 2) &  # Current temp near or below freezing
            (df['precipitation'] > 0)   # Precipitation present
        )
        df['snow_probability'] = snow_conditions.rolling(
            window=3, min_periods=1
        ).mean()
        
        # b) Melt Conditions
        # Consider melt conditions when temp rises above freezing
        df['temp_rising'] = (df['temp_change'] > 0).astype(int)
        
        df['melt_potential'] = np.where(
            (df['temperature'] > 0) &    # Current temp above freezing
            (df['temp_rising']),         # Temperature is rising
            df['temperature'],           # Use temperature as melt potential
            0
        )
        
        # 5. Additional Temporal Features
        # Seasonal decomposition proxy (30-day rolling statistics)
        df['temp_30d_mean'] = df['temperature'].rolling(window=30, min_periods=1).mean()
        df['temp_seasonal_dev'] = df['temperature'] - df['temp_30d_mean']
        
        # Drop intermediate columns
        df = df.drop(['temp_change', 'significant_temp_change', 'temp_rising'], axis=1)
        
        return df
    
    @staticmethod
    def _calculate_persistence(series: pd.Series) -> pd.Series:
        """Calculate length of consecutive occurrences."""
        spell_length = series.groupby((series != series.shift()).cumsum()).cumcount() + 1
        return np.where(series == 1, spell_length, 0)
    
    def process_basin(self, basin_id: str) -> pd.DataFrame:
        """
        Process temperature metrics for a specific basin.
        
        Args:
            basin_id: CAMELS-GB basin ID
            
        Returns:
            DataFrame with original and new metrics
        """
        # Load basin data
        file_pattern = f"CAMELS_GB_hydromet_timeseries_{basin_id}_*.csv"
        try:
            basin_file = next(self.data_dir.glob(file_pattern))
        except StopIteration:
            raise FileNotFoundError(f"No data file found for basin {basin_id}")
        
        # Read and process data
        df = pd.read_csv(basin_file, parse_dates=['date'], index_col='date')
        df_processed = self.calculate_temp_metrics(df)
        
        return df_processed
    
    def save_processed_data(self, basin_id: str, output_dir: str) -> None:
        """
        Save processed data for a basin.
        
        Args:
            basin_id: CAMELS-GB basin ID
            output_dir: Directory to save processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            df_processed = self.process_basin(basin_id)
            output_file = output_path / f"CAMELS_GB_temp_metrics_{basin_id}.csv"
            df_processed.to_csv(output_file)
            logger.info(f"Processed and saved data for basin {basin_id}")
        except Exception as e:
            logger.error(f"Failed to process basin {basin_id}: {str(e)}")

def process_all_basins(data_dir: str, output_dir: str) -> None:
    """
    Process temperature metrics for all CAMELS-GB basins.
    
    Args:
        data_dir: Path to CAMELS-GB timeseries data directory
        output_dir: Directory to save processed data
    """
    processor = TemperatureMetricsProcessor(data_dir)
    data_dir_path = Path(data_dir)
    
    # Get all unique basin IDs from filenames
    basin_files = list(data_dir_path.glob("CAMELS_GB_hydromet_timeseries_*.csv"))
    if not basin_files:
        raise FileNotFoundError(f"No CAMELS-GB timeseries files found in {data_dir}")
    
    total_basins = len(basin_files)
    logger.info(f"Found {total_basins} basins to process")
    
    for i, basin_file in enumerate(basin_files, 1):
        # Extract basin ID from filename
        basin_id = basin_file.stem.split("_")[4]
        
        try:
            processor.save_processed_data(basin_id, output_dir)
            logger.info(f"Processed basin {basin_id} ({i}/{total_basins})")
        except Exception as e:
            logger.error(f"Error processing basin {basin_id}: {str(e)}")
            continue

def main():
    """Main execution function."""
    # Initialize paths
    data_dir = "/Users/matviikotolyk/hydrology_data/CAMELS_GB/timeseries"
    output_dir = "/Users/matviikotolyk/hydrology_data/CAMELS_GB/processed"
    
    logger.info("Starting temperature metrics processing for all basins")
    process_all_basins(data_dir, output_dir)
    logger.info("Completed processing all basins")

if __name__ == "__main__":
    main()