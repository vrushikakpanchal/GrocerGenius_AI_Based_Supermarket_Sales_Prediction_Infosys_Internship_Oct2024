# Import required libraries
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import r2_score
from PIL import Image

# Load saved encoders and model for inference
ordinal_encoder = joblib.load('app_files/ordinal_encoder.pkl')
onehot_encoder = joblib.load('app_files/onehot_encoder.pkl')
scaler = joblib.load('app_files/standard_scaler.pkl')

# Load label encoders for 'Item_Identifier' and 'Outlet_Identifier'
item_identifier_encoder = joblib.load('app_files/Item_Identifier_label_encoder.pkl')
outlet_identifier_encoder = joblib.load('app_files/Outlet_Identifier_label_encoder.pkl')

# Load the saved Gradient Boosting (XGBoost) model for predictions
best_gb_model = joblib.load('app_files/xgboost_model.pkl')

# Define data preprocessing function for inference
def data_preprocessing(input_data):
    # Handling missing values (same logic as during training)
    input_data['Item_Weight'] = input_data['Item_Weight'].fillna(input_data.groupby('Item_Type')['Item_Weight'].transform('mean'))
    input_data['Outlet_Size'] = input_data['Outlet_Size'].fillna(input_data['Outlet_Size'].mode()[0])

    # Feature derivation (same logic as training phase)
    input_data['Outlet_Age'] = 2024 - input_data['Outlet_Establishment_Year']
    input_data['Price_Per_Unit_Weight'] = input_data['Item_MRP'] / input_data['Item_Weight']

    # Simplify Item_Fat_Content (same logic as training)
    input_data['Item_Fat_Content'] = input_data['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})

    # Log Transformation for Item Visibility (same as training)
    input_data['Item_Visibility_Log'] = np.log1p(input_data['Item_Visibility'])

    # MRP Categorization (same as training)
    min_value = input_data['Item_MRP'].min()
    max_value = input_data['Item_MRP'].max()
    range_value = max_value - min_value
    input_data['MRP_Tier'] = input_data['Item_MRP'].apply(lambda x: 'Low' if x <= min_value + 0.33 * range_value else
                                                          'Medium' if x <= min_value + 0.66 * range_value else 'High')

    # Encoding the data (scaling and transforming)
    numeric_features = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 'Price_Per_Unit_Weight']
    ordinal_features = ['Outlet_Size', 'MRP_Tier']
    nominal_features = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type']
    
    # Scale numeric features
    input_data[numeric_features] = scaler.transform(input_data[numeric_features])
    
    # Encode ordinal features
    if ordinal_features:
        input_data[ordinal_features] = ordinal_encoder.transform(input_data[ordinal_features])

    # One-hot encode nominal features
    if nominal_features:
        nominal_encoded = onehot_encoder.transform(input_data[nominal_features])
        nominal_cols = onehot_encoder.get_feature_names_out(nominal_features)
        nominal_df = pd.DataFrame(nominal_encoded, columns=nominal_cols)
        input_data = pd.concat([input_data, nominal_df], axis=1)
        input_data.drop(columns=nominal_features, inplace=True)

    # Label encode ID features (if necessary)
    input_data['Item_Identifier'] = item_identifier_encoder.transform(input_data['Item_Identifier'])
    input_data['Outlet_Identifier'] = outlet_identifier_encoder.transform(input_data['Outlet_Identifier'])

    return input_data

# Streamlit UI for user input
def main():
    st.set_page_config(
        page_title='GrocerGenius: AI-Powered Sales Forecasting for Supermarkets',
        page_icon='🛒',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.markdown("""<style>
        body { background-color: #FBF6E9; }
        .main { background-color: #E3F0AF; color: #118B50; }
    </style>""", unsafe_allow_html=True)

    st.sidebar.title('Navigation')
    nav_option = st.sidebar.radio("Go to", ['Home', 'Test Us', 'About Us'])

    if nav_option == 'Home':
        st.title('🛒 GrocerGenius: AI-Powered Sales Forecasting for Supermarkets')
        col1, col2 = st.columns(2)  # Equal division of columns

        with col1:
            # Add the image
            image = Image.open('app_files/images/img.jpeg')
            resized_image = image.resize((600,430))  # Adjust size for better fit
            st.image(resized_image, use_container_width=True)

        with col2:
            st.markdown(
        """
<div style="border-radius: 12px; background-color: rgb(240, 255, 220); padding: 15px; margin-bottom: 10px;">
    <h4 style="color: #355F2E; font-weight: bold;">🌟 Why GrocerGenius?</h4>
    <ul style="color: #355F2E; margin-left: 15px; padding-left: 15px;">
        <li>Cut inventory waste with accurate forecasts.</li>
        <li>Boost profits by optimizing stock levels.</li>
        <li>Stay ahead with insights tailored to your business.</li>
    </ul>
</div>
<div style="border-radius: 12px; background-color: rgb(240, 255, 220); padding: 15px;">
    <h4 style="color: #355F2E; font-weight: bold;">📊 Smart Predictions, Smarter Profits</h4>
    <ul style="color: #355F2E; margin-left: 15px; padding-left: 15px;">
        <li>Harness the power of XGBoost, a gradient boosting approach, to deliver accurate and reliable predictions.</li>
        <li>Seamlessly input details like product visibility, pricing, and store characteristics for tailored insights.</li>
        <li>Maximize profitability with data-driven decisions designed to streamline your operations.</li>
    </ul>
</div>
""",
        unsafe_allow_html=True)

    elif nav_option == 'Test Us':
        st.title('📊 Predict Sales')
        col1, col2 = st.columns(2)

        with col1:
            st.header('📦 Product Information')
            item_identifier = st.selectbox('Item Identifier', options=sorted(['FDN15', 'FDX07', 'FDA15', 'DRC01', 'NCD19']), index=0)
            item_weight = st.number_input('Item Weight (in kg)', min_value=0.0, max_value=100.0, value=9.3, help='Weight of the product.')
            item_fat_content = st.selectbox('Item Fat Content', options=['Low Fat', 'Regular'], index=0, help='Fat content of the product.')
            item_visibility = st.number_input('Item Visibility', min_value=0.0, max_value=0.33, value=0.016, step=0.001, help='Visibility of the product in the store')
            item_type = st.selectbox('Item Type', options=sorted(['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood']), index=4, help='Category of the product.')
            item_mrp = st.number_input('Item MRP', min_value=0.0, max_value=500.0, value=249.81, step=0.01, help='Maximum Retail Price of the product.')

        with col2:
            st.header('🏬 Store Information')
            outlet_identifier = st.selectbox('Outlet Identifier', options=sorted(['OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027', 'OUT045']), index=0)
            outlet_establishment_year = st.number_input('Outlet Establishment Year', min_value=1980, max_value=2020, value=1999, step=1, help='Year the store was established.')
            outlet_size = st.selectbox('Outlet Size', options=['Small', 'Medium', 'High'], index=1, help='Size of the store.')
            outlet_location_type = st.selectbox('Outlet Location Type', options=['Tier 1', 'Tier 2', 'Tier 3'], index=0, help='Location type of the store.')
            outlet_type = st.selectbox('Outlet Type', options=['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'], index=0, help='Type of store.')

        # Custom button styling with a larger font size, green background, and white text
        button_style = """
            <style>
                .stButton button {
                   background-color: green;
                   color: white;
                   font-size: 18px;
                   font-weight: bold;
                }
            </style>
        """
        st.markdown(button_style, unsafe_allow_html=True)

        # Custom box styling for prediction results
        box_style = """
            <style>
                .prediction-box {
                background-color: rgb(240, 255, 220);  /* Light green background */
                padding: 20px;
                border-radius: 10px;
                font-size: 20px;
                font-weight: bold;
                color: #355F2E;  /* Dark green text color */
                border: 2px solid #00796b;
                }
            </style>
        """
        st.markdown(box_style, unsafe_allow_html=True)

        # Function to predict and display sales
        if st.button('Predict Sales'):
            # Collecting user inputs
            input_data = pd.DataFrame({
                'Item_Identifier': [item_identifier],
                'Item_Weight': [item_weight],
                'Item_Fat_Content': [item_fat_content],
                'Item_Visibility': [item_visibility],
                'Item_Type': [item_type],
                'Item_MRP': [item_mrp],
                'Outlet_Identifier': [outlet_identifier],
                'Outlet_Establishment_Year': [outlet_establishment_year],
                'Outlet_Size': [outlet_size],
                'Outlet_Location_Type': [outlet_location_type],
                'Outlet_Type': [outlet_type]
            })
            
            # Preprocess the data
            preprocessed_data = data_preprocessing(input_data)

            # Make a prediction using the trained model
            predicted_sales = best_gb_model.predict(preprocessed_data)

            # Display the prediction
            st.markdown(f"""
                <div class="prediction-box">
                     <h5><strong>Predicted Sales: ₹ {predicted_sales[0]:,.2f}</strong></h5>
                     <h5><strong>Item Identifier: {item_identifier}</strong></h5>
                     <h5><strong>Outlet Identifier: {outlet_identifier}</strong></h5>
                </div>
            """, unsafe_allow_html=True)

    elif nav_option == 'About Us':
        # Title
        st.title('About GrocerGenius')

        # Project Overview
        st.write("""
        **GrocerGenius** is an AI-powered tool designed to optimize retail operations by providing accurate sales forecasts. 
        It helps businesses reduce waste, manage inventory, and boost profitability with the power of machine learning.
        """)

        # Add the image with resized dimensions
        pipeline = Image.open('app_files/images/pipeline.png')

        # Resize the image with slightly smaller width and height
        resized_image = pipeline.resize((1000, 320))  # Adjusted size (width, height) for a better fit
        st.image(resized_image, caption="Machine Learning Pipeline", use_container_width=True, width=600)

        # Mentorship and Internship Experience
        st.write("""
    **Mentorship:** This project was developed under the guidance of **Mr. AMAL SALILAN**, whose dedication and insightful mentorship provided us with a clear roadmap, empowering us to turn GrocerGenius into a successful AI solution.
    
    It was a great experience to work with Infosys Springboard for the **Internship 5.0**, where I gained hands-on experience in developing and deploying AI solutions, which has been truly enriching.
    """)

    # Personal Introduction and Contact Information
        st.write("""
    I'm **Vrushika K Panchal**, who worked as an AI - Data Science Domain Intern.
    
    Feel free to connect with me:
    - GitHub: [https://github.com/vrushikakpanchal](https://github.com/vrushikakpanchal)
    - LinkedIn: [https://linkedin.com/in/vrushikakpanchal](https://linkedin.com/in/vrushikakpanchal)
    - Email: [vrushikakpanchal@gmail.com](mailto:vrushikakpanchal@gmail.com)
    """)
        
# Run the app
if __name__ == '__main__':
    main()
