import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib # To save and load the model and encoder

def train_and_save_model():
    """
    Loads data, preprocesses it, trains a RandomForestClassifier,
    and saves the trained pipeline and LabelEncoder.
    """
    try:
        df = pd.read_csv('adult 3.csv')
        print("Dataset loaded successfully for training.")
    except FileNotFoundError:
        print("Error: 'adult 3.csv' not found. Please ensure the file is in the correct directory.")
        return

    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Define columns
    numerical_cols = ['age', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
    target_col = 'income'

    # Impute missing values (before splitting to ensure consistent imputation for all data)
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode(), inplace=True) # Use  as mode() can return multiple values

    # Encode the target variable
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

    # Define features (X) and target (y)
    X = df.drop(columns=[target_col, 'fnlwgt']) # Drop fnlwgt as it's a sample weight
    y = df[target_col]

    # Create a column transformer for preprocessing
    # Numerical features will be scaled
    # Categorical features will be one-hot encoded
    preprocessor = ColumnTransformer(
        transformers=)

    # Create a pipeline that first preprocesses the data and then trains the model
    model_pipeline = Pipeline(steps=)

    # Train the model on the full dataset (for deployment, you'd train on training set)
    print("Training the model...")
    model_pipeline.fit(X, y)
    print("Model training complete.")

    # Save the trained pipeline and LabelEncoder
    joblib.dump(model_pipeline, 'salary_prediction_model.pkl')
    joblib.dump(le, 'label_encoder.pkl')

    # Save unique values for selectboxes in Streamlit
    categorical_options = {col: df[col].unique().tolist() for col in categorical_cols}
    joblib.dump(categorical_options, 'categorical_options.pkl')

    print("Model, LabelEncoder, and categorical options saved successfully.")

if __name__ == "__main__":
    train_and_save_model()