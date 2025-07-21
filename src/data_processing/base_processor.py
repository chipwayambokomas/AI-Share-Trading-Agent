import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseProcessor(ABC):
    """Abstract Base Class for data processors."""
    
    def __init__(self, settings):
        self.settings = settings
        self.mode = settings.PREDICTION_MODE

    @abstractmethod
    def process(self, df: pd.DataFrame):
        """The main method to process data."""
        pass

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shared data cleaning logic."""
        logger.info("Starting data cleaning process...")
        cleaned_stocks = []
        
        # Convert values in feature columns to numeric and set non-positive values to NaN
        for col in self.settings.FEATURE_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df.loc[df[col] <= 0, col] = np.nan
        
        initial_stock_count = df['StockID'].nunique()
        
        # Process each stock individually
        for stock_id, stock_df in df.groupby('StockID'):
            original_len = len(stock_df)
            # Calculate missing target values
            missing_count = stock_df[self.settings.TARGET_COLUMN].isnull().sum()
            missing_pct = missing_count / original_len if original_len > 0 else 0
            
            # Exclude stocks with more than 10% missing target data
            if missing_pct > 0.10:
                logger.warning(f"Excluding stock '{stock_id}': {missing_pct:.1%} missing target data.")
                continue
                
            cleaned_stock_df = stock_df.copy()
            # Forward fill missing feature values
            cleaned_stock_df[self.settings.FEATURE_COLUMNS] = cleaned_stock_df[self.settings.FEATURE_COLUMNS].ffill()
            # Drop rows where features are still missing
            cleaned_stock_df.dropna(subset=self.settings.FEATURE_COLUMNS, inplace=True)
            
            # Exclude stocks that are empty after cleaning
            if cleaned_stock_df.empty:
                logger.warning(f"Excluding stock '{stock_id}': Empty after cleaning.")
                continue
                
            cleaned_stocks.append(cleaned_stock_df)
            
        # Raise error if no stocks remain after cleaning
        if not cleaned_stocks:
            raise ValueError("No stocks remained after data cleaning. Check data quality.")
            
        # Concatenate all cleaned stocks into a single DataFrame
        final_df = pd.concat(cleaned_stocks, ignore_index=True)
        final_stock_count = final_df['StockID'].nunique()
        
        logger.info(f"Data cleaning complete. Kept {final_stock_count}/{initial_stock_count} stocks.")
        return final_df