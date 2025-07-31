import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('feature-engineer')

def create_feature(df):
    logger.info("Creating features from the dataset")
    
    df_featured = df.copy()
    
    # Calculate house age
    current_year = datetime.now().year
    df_featured['house_age'] = current_year - df_featured['year_built']
    logger.info("Created 'house_age' feature")
    
    # Price per square foot
    df_featured['price_per_sqft'] = df_featured['price'] / df_featured['sqft']
    logger.info("Created 'price_per_sqft' feature")
    
    # Bedroom to bathroom ratio
    df_featured['bed_bath_ratio'] = df_featured['bedrooms'] / df_featured['bathrooms']
    # Handle division by zero
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].replace([np.inf, -np.inf], np.nan)
    df_featured['bed_bath_ratio'] = df_featured['bed_bath_ratio'].fillna(0)
    logger.info("Created 'bed_bath_ratio' feature")
    
    return df_featured

def create_preprocessor():
    logger.info("Creating preprocessor for feature engineering")
    
    categorical_features = ['location', 'condition']
    numerical_features = ['sqft', 'bedrooms', 'bathrooms', 'house_age', 'price_per_sqft', 'bed_bath_ratio']
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers = [
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def run_feature_engineering(input_file, output_file, preprocessor_file):
    logger.info(f"Running feature engineering on {input_file}")
    df = pd.read_csv(input_file)
    
    # create features
    df_featured = create_feature(df)
    logger.info(f"Features created with shape: {df_featured.shape}")
    
    # create and fit preprocessor
    preprocessor = create_preprocessor()
    X = df_featured.drop(columns=['price'])
    y = df_featured['price']
    X_transformed = preprocessor.fit_transform(X)
    logger.info("Preprocessing completed")
    
    joblib.dump(preprocessor, preprocessor_file)
    logger.info(f"Preprocessor saved to {preprocessor_file}")
    
    # save the transformed data
    df_transformed = pd.DataFrame(X_transformed)
    if y is not None:
        df_transformed['price'] = y.values
    df_transformed.to_csv(output_file, index=False)
    logger.info(f"Transformed data saved to {output_file}")
    
    return df_transformed


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run feature engineering on the dataset.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the transformed CSV file')
    parser.add_argument('--preprocessor_file', type=str, required=True, help='Path to save the preprocessor file')
    
    args = parser.parse_args()
    
    run_feature_engineering(args.input_file, args.output_file, args.preprocessor_file)