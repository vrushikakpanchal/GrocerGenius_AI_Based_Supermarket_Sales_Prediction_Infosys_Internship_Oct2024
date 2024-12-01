# Grocery Sales Prediction

## Introduction

Welcome to the **Grocery Sales Prediction** project! This initiative aims to forecast the sales of grocery items using advanced machine learning techniques. By analyzing historical sales data, we empower retailers and stakeholders to make informed decisions on inventory management, pricing strategies, and marketing campaigns. Our goal is to enhance operational efficiency and maximize profitability in the competitive grocery market.

## Technology Stack

- **Programming Language**: Python 3.x
- **Data Manipulation**: Pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Model Persistence**: Joblib
- **Web Framework**: Streamlit
- **Version Control**: Git
- **Development Environment**: Jupyter Notebooks

## Workflow

1. **Data Collection**: Gather raw sales data from various grocery outlets and consolidate it into a structured format.
2. **Data Preprocessing**: Clean and preprocess the data to handle missing values, outliers, and inconsistent formats.
3. **Feature Engineering**: Create new features and transform existing ones to improve model performance.
4. **Model Training**: Use machine learning algorithms to train predictive models on the processed data.
5. **Model Evaluation**: Assess model accuracy using appropriate metrics and fine-tune hyperparameters.
6. **Model Deployment**: Deploy the trained model using Streamlit for real-time user interaction.
7. **Inferencing**: Allow users to input new data and obtain sales predictions through the deployed model.

## Preprocessing Methods

Our preprocessing pipeline includes the following steps:

- **Handling Missing Values**:
  - *Item Weight*: Imputed using the median weight grouped by item type.
  - *Outlet Size*: Filled using the mode of the outlet size grouped by outlet type.
- **Outlier Detection and Treatment**:
  - Used the Interquartile Range (IQR) method to cap outliers in numerical features.
- **Data Standardization**:
  - Standardized categories in *Item Fat Content* to unify similar labels.
- **Feature Creation**:
  - *Item Visibility Bins*: Categorized item visibility into 'Low', 'Medium', and 'High'.
  - *Years Since Establishment*: Calculated the number of years each outlet has been operational.
- **Encoding Categorical Variables**:
  - *One-Hot Encoding*: Applied to nominal variables like *Item Type* and *Outlet Type*.
  - *Ordinal Encoding*: Used for ordinal variables like *Outlet Size* and *Outlet Location Type*.
  - *Mean Target Encoding*: Implemented for *Outlet Identifier* using mean sales.
- **Feature Transformation**:
  - Applied log transformation to *Item Visibility* to reduce skewness.

## Modeling

- **Algorithm Used**: Random Forest Regressor
- **Justification**:
  - Handles non-linear data effectively.
  - Robust to overfitting due to ensemble nature.
  - Capable of capturing feature interactions.
- **Model Training**:
  - Split data into training and test sets.
  - Trained on preprocessed features and target variable.
- **Hyperparameter Tuning**:
  - Performed grid search to find the optimal parameters.
- **Evaluation Metrics**:
  - *Mean Squared Error (MSE)*: Measures average squared difference between predicted and actual values.
  - *R-squared (R²)*: Indicates the proportion of variance explained by the model.

## Inferencing

- **User Interface**: Built with Streamlit to provide an interactive web app.
- **Functionality**:
  - Users can input feature values through dropdowns, sliders, and input fields.
  - Real-time predictions are displayed upon submission.
- **Error Handling**:
  - Implemented try-except blocks to capture and display errors gracefully.
- **Model Integration**:
  - Loaded the trained model and encoders to process user inputs and generate predictions.

## Usage

### Prerequisites

- **Python 3.x**
- **Libraries**:
  - pandas
  - numpy
  - scikit-learn
  - joblib
  - streamlit

### Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/grocery_sales_prediction.git
   cd grocery_sales_prediction
   ```

2. **Create a Virtual Environment**

   ```bash
   python3 -m venv env
   source env/bin/activate  # For Windows: env\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Directory Structure**

   Ensure your directory structure matches the following:

   ```
   grocery_sales_prediction/
   ├── data_alchemy/
   │   ├── raw/
   │   │   └── train.csv
   ├── model_factory/
   │   ├── models/
   │   ├── encoders/
   │   └── features/
   ├── codebase/
   │   ├── utils.py
   │   ├── training_script.py
   │   └── app.py
   └── README.md
   ```

5. **Place Your Data**

   - Copy your `train.csv` file into `data_alchemy/raw/`.

6. **Train the Model**

   ```bash
   cd codebase
   python training_script.py
   ```

7. **Run the Streamlit App**

   ```bash
   streamlit run app.py
   ```

8. **Access the Application**

   - Open your web browser and navigate to `http://localhost:8501`.

### Using the Application

- **Input Features**: Enter the required information such as item details and outlet characteristics.
- **Predict Sales**: Click the "Predict" button to obtain the estimated sales.
- **View Results**: The predicted sales will be displayed on the screen.

## Contributors

We extend our heartfelt gratitude to the following contributors who have been instrumental in the success of this project:

- **Amal Salilan** (@amalsalilan) - *Mentor*
- **Aman** (@theamansyed)
- **Vrushika K Panchal** (@vrushika-k-panchal)
- **Chetan** (@Chetanp717)
- **Rimi** (@rs2103)
- **Shilpa Manaji** (@Shilpa-Manaji)
- **Tharun** (@Kottetharun-09)
- **Sumithra** (@Sumithra-git)
- **Yanvi Arora** (@YanviAroraCS)
- **Sayantan** (@SayanRony)
- **Muskan Asthana** (@muskan42)
- **Purnima Pattnaik** (@Purnima07-sudo)
- **Rameswar Bisoyi** (@RB137)
- **Raunit** (@raunit45)
- **Hima Mankanta** (@manu-vasamsetti)
- **Nuka Abhinay** (@NUKA-ABHINAY)
- **Anjan Kumar** (@Anjankumarkamalapur)

*Thank you for your dedication, expertise, and collaborative spirit.*

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **Mentor**: Amal Salilan (@amalsalilan) 

- **Community**: Thanks to the open-source community for providing tools and resources.

---

*For any queries or contributions, please contact us at [your-email@example.com](mailto:your-email@example.com).*
