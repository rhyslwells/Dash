"""Data loading and preparation module"""

import pandas as pd
from config import DATA_DIR


def load_data():
    """Load poverty data from CSV files"""
    poverty_data = pd.read_csv(DATA_DIR / 'PovStatsData.csv')
    poverty = pd.read_csv(DATA_DIR / 'poverty.csv', low_memory=False)
    return poverty_data, poverty


def prepare_data(poverty):
    """Prepare data for logistic regression model"""
    # Get Gini data
    gini_data = poverty.dropna(subset=['GINI index (World Bank estimate)']).copy()
    
    # Extract relevant features
    features_to_use = ['GINI index (World Bank estimate)']
    
    # Get additional features if available
    income_cols = [col for col in gini_data.columns if 'Income share' in col]
    features_to_use.extend(income_cols[:3])  # Use first 3 income share columns
    
    # Prepare dataset
    df = gini_data[['Country Name', 'year'] + features_to_use].dropna()
    
    if df.empty or len(df) < 10:
        # Fallback: use Gini data only
        df = gini_data[['Country Name', 'year', 'GINI index (World Bank estimate)']].dropna()
        features = ['GINI index (World Bank estimate)']
    else:
        features = features_to_use
    
    # Create binary target: high poverty (GINI > median)
    gini_col = 'GINI index (World Bank estimate)'
    df['high_poverty'] = (df[gini_col] > df[gini_col].median()).astype(int)
    
    return df, features, gini_col
