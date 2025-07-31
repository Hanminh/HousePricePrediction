import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger('data-processor')

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    logger.info('Cleaning data...')
    df_cleaned = df.copy()
    
    # handle missing values
    for column in df_cleaned.columns:
        missing_count = df_cleaned[column].isnull().sum()
        if missing_count > 0:
            logger.warning(f"Column '{column}' has {missing_count} missing values.")
            
            # for numerical columns, fill with mean
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                mean_value = df_cleaned[column].mean()
                df_cleaned[column].fillna(mean_value, inplace=True)
                logger.info(f"Filled missing values in '{column}' with mean: {mean_value}")
            # for categorical columns, fill with mode
            else:
                mode_value = df_cleaned[column].mode()[0]
                df_cleaned[column].fillna(mode_value, inplace=True)
                logger.info(f"Filled missing values in '{column}' with mode: {mode_value}")
                
    Q1 = df_cleaned['price'].quantile(0.25)
    Q3 = df_cleaned['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outlier = df_cleaned[(df_cleaned['price'] < lower_bound) | (df_cleaned['price'] > upper_bound)]
    
    if not outlier.empty:
        logger.warning(f"Outliers detected in the following columns: {outlier.columns.tolist()}")
        df_cleaned = df_cleaned[(df_cleaned['price'] >= lower_bound) & (df_cleaned['price'] <= upper_bound)]
        logger.info("Outliers removed from the dataset.")
        
    return df_cleaned

def process_data(input_file, output_file):
    output_path = Path(output_file).parent
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = load_data(input_file)
    logger.info(f"Data loaded with shape: {df.shape}")
    df_cleaned = clean_data(df)
    logger.info(f"Data cleaned with shape: {df_cleaned.shape}")
    # save the cleaned data
    df_cleaned.to_csv(output_file, index=False)
    logger.info(f"Cleaned data saved to {output_file}")
    return df_cleaned

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process and clean data.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the cleaned CSV file')
    
    args = parser.parse_args()
    
    processed_data = process_data(args.input_file, args.output_file)
    logger.info("Data processing completed successfully.")