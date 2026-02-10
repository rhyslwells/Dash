"""Configuration and constants for the Dash app"""

from pathlib import Path

# Determine the data directory path
current_file = Path(__file__).resolve()
project_root = current_file.parent
DATA_DIR = project_root / 'data'

# Define regions to exclude from analysis
REGIONS_TO_EXCLUDE = [
    'East Asia & Pacific',
    'Europe & Central Asia',
    'Fragile and conflict affected situations',
    'High income',
    'IDA countries classified as fragile situations',
    'IDA total',
    'Latin America & Caribbean',
    'Low & middle income',
    'Low income',
    'Lower middle income',
    'Middle East & North Africa',
    'Middle income',
    'South Asia',
    'Sub-Saharan Africa',
    'Upper middle income',
    'World'
]

# Model configuration
MODEL_RANDOM_STATE = 42
MODEL_MAX_ITER = 1000
TEST_SET_SIZE = 0.3
