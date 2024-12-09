# codebase/utils.py

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Define base directories relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model_factory')
ENCODER_DIR = os.path.join(MODEL_DIR, 'encoders')
FEATURE_DIR = os.path.join(MODEL_DIR, 'features')
MODEL_FILE = os.path.join(MODEL_DIR, 'models', 'random_forest_model.pkl')

def preprocess_data(df, is_training=True):
    data = df.copy()

    # Drop unnecessary columns
    data.drop(columns=['Item_Identifier', 'Item_Outlet_Sales'], errors='ignore', inplace=True)

    # Handle outliers
    for column in data.select_dtypes(include='number').columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = data[column].apply(
            lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
        )

    # Fill missing values
    data['Item_Weight'] = data.groupby('Item_Type')['Item_Weight'].transform(
        lambda x: x.fillna(x.median())
    )
    data['Outlet_Size'] = data.groupby('Outlet_Type')['Outlet_Size'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Medium')
    )

    # Standardize 'Item_Fat_Content'
    data['Item_Fat_Content'] = data['Item_Fat_Content'].replace(
        {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
    )

    # Create 'Years_Since_Establishment'
    data['Years_Since_Establishment'] = 2024 - data['Outlet_Establishment_Year']

    # Create 'Item_Visibility_Bins' using consistent bins
    if is_training:
        max_visibility = data['Item_Visibility'].max()
        bins = [-0.001, 0.05, 0.15, max_visibility + 0.001]
        labels = ['Low', 'Medium', 'High']
        # Save bins
        bin_info = {'bins': bins, 'labels': labels}
        os.makedirs(ENCODER_DIR, exist_ok=True)
        joblib.dump(bin_info, os.path.join(ENCODER_DIR, 'visibility_bins.pkl'))
    else:
        # Load bins
        bin_info = joblib.load(os.path.join(ENCODER_DIR, 'visibility_bins.pkl'))
        bins = bin_info['bins']
        labels = bin_info['labels']

    data['Item_Visibility_Bins'] = pd.cut(
        data['Item_Visibility'],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # Encode 'Outlet_Identifier' using mean target encoding
    if is_training:
        # Compute mean 'Item_Outlet_Sales' for each 'Outlet_Identifier'
        outlet_sales_mapping = df.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean().to_dict()
        # Save the mapping
        joblib.dump(outlet_sales_mapping, os.path.join(ENCODER_DIR, 'outlet_sales_mapping.pkl'))
        # Save global mean sales
        global_mean_sales = df['Item_Outlet_Sales'].mean()
        joblib.dump(global_mean_sales, os.path.join(ENCODER_DIR, 'global_mean_sales.pkl'))
    else:
        # Load the mapping
        outlet_sales_mapping = joblib.load(os.path.join(ENCODER_DIR, 'outlet_sales_mapping.pkl'))
        # Load global mean sales
        global_mean_sales = joblib.load(os.path.join(ENCODER_DIR, 'global_mean_sales.pkl'))

    # Map 'Outlet_Identifier' to mean sales
    data['Outlet_Identifier_Mean_Sales'] = data['Outlet_Identifier'].map(outlet_sales_mapping)
    # For unknown 'Outlet_Identifier's, fill with global mean
    data['Outlet_Identifier_Mean_Sales'].fillna(global_mean_sales, inplace=True)
    # Drop 'Outlet_Identifier' column
    data.drop('Outlet_Identifier', axis=1, inplace=True)

    # Encoding
    nominal_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Type']
    ordinal_columns = ['Item_Visibility_Bins', 'Outlet_Size', 'Outlet_Location_Type']

    if is_training:
        # One-Hot Encoding for nominal columns
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        encoded_nominal = ohe.fit_transform(data[nominal_columns])
        joblib.dump(ohe, os.path.join(ENCODER_DIR, 'ohe_encoder.pkl'))
    else:
        # Load the saved OneHotEncoder
        ohe = joblib.load(os.path.join(ENCODER_DIR, 'ohe_encoder.pkl'))
        encoded_nominal = ohe.transform(data[nominal_columns])

    encoded_nominal_df = pd.DataFrame(
        encoded_nominal, columns=ohe.get_feature_names_out(nominal_columns)
    )
    data.reset_index(drop=True, inplace=True)
    encoded_nominal_df.reset_index(drop=True, inplace=True)
    data = pd.concat([data, encoded_nominal_df], axis=1)
    data.drop(nominal_columns, axis=1, inplace=True)

    # Ordinal Encoding for ordinal columns
    ordinal_categories = [
        ['Low', 'Medium', 'High'],
        ['Small', 'Medium', 'High'],
        ['Tier 1', 'Tier 2', 'Tier 3']
    ]
    if is_training:
        ordinal_encoder = OrdinalEncoder(
            categories=ordinal_categories, handle_unknown='use_encoded_value', unknown_value=-1
        )
        data[ordinal_columns] = ordinal_encoder.fit_transform(data[ordinal_columns])
        joblib.dump(ordinal_encoder, os.path.join(ENCODER_DIR, 'ordinal_encoder.pkl'))
    else:
        ordinal_encoder = joblib.load(os.path.join(ENCODER_DIR, 'ordinal_encoder.pkl'))
        data[ordinal_columns] = ordinal_encoder.transform(data[ordinal_columns])

    # Log transformation of 'Item_Visibility'
    data['Item_Visibility_Log'] = np.log1p(data['Item_Visibility'])
    data.drop('Item_Visibility', axis=1, inplace=True)

    # Drop 'Outlet_Establishment_Year' as it's not needed after creating 'Years_Since_Establishment'
    data.drop('Outlet_Establishment_Year', axis=1, inplace=True)

    return data

def train_model(data):
    # Prepare features and target
    y = data['Item_Outlet_Sales']
    X = data.drop(columns=['Item_Outlet_Sales'], errors='ignore')

    # Save feature names
    feature_names = X.columns.tolist()
    os.makedirs(FEATURE_DIR, exist_ok=True)
    joblib.dump(feature_names, os.path.join(FEATURE_DIR, 'feature_names.pkl'))

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Train the models
    print("Optimised Model training started.")
    xgb_model2=XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42)
    xgb_model2.fit(X_train, y_train)
    rf_model = RandomForestRegressor()
    rf_model.fit(X_train, y_train)
    y_pred = xgb_model2.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Optimised Model training completed.")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    os.makedirs(os.path.join(MODEL_DIR, 'models'), exist_ok=True)
    joblib.dump(rf_model, MODEL_FILE)
    print("Model saved.")

    # Save the trained model
    os.makedirs(os.path.join(MODEL_DIR, 'models'), exist_ok=True)
    joblib.dump(rf_model, MODEL_FILE)

    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model training completed.")
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    return rf_model

def run_inference(input_data):
    # Preprocess data
    preprocessed_data = preprocess_data(input_data, is_training=False)

    # Load feature names
    feature_names = joblib.load(os.path.join(FEATURE_DIR, 'feature_names.pkl'))

    # Ensure all expected features are present in preprocessed_data
    for feature in feature_names:
        if feature not in preprocessed_data.columns:
            preprocessed_data[feature] = 0  # Assign a default value (e.g., 0)

    # Reorder columns to match the training data
    preprocessed_data = preprocessed_data[feature_names]

    # Load the trained model
    rf_model = joblib.load(MODEL_FILE)

    # Make predictions
    predictions = rf_model.predict(preprocessed_data)

    # Return predictions
    return predictions
