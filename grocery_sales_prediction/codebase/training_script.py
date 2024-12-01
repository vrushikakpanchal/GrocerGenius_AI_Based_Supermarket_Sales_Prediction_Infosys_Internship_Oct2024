# codebase/training_script.py

import os
import pandas as pd
from utils import preprocess_data, train_model

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data_alchemy', 'raw')

# Load your training data
df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))

# Preprocess and train
preprocessed_data = preprocess_data(df, is_training=True)
preprocessed_data['Item_Outlet_Sales'] = df['Item_Outlet_Sales'].values
model = train_model(preprocessed_data)
