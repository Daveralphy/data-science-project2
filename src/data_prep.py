import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# --- Configuration ---
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_PATH = "data/raw/telco_churn.csv"
PROCESSED_DATA_PATH = "data/processed/"
MODELS_DIR = "models/"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")


def create_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(url: str, path: str) -> pd.DataFrame:
    """Loads data from URL and saves it locally, or loads from local cache."""
    if not os.path.exists(path):
        print(f"Downloading data from {url}...")
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
        print(f"Data saved to {path}")
    else:
        print(f"Loading data from local cache: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values and corrects data types."""
    # Convert TotalCharges to numeric, coercing errors to NaN
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Impute missing TotalCharges with 0 for new customers (tenure=0)
    # This is a reasonable business assumption.
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features as specified in the project brief."""
    # tenure_bucket
    bins = [-1, 6, 12, 24, 73] # Use -1 to include 0
    labels = ['0-6m', '7-12m', '13-24m', '24m+']
    df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels)

    # services_count
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    # Convert 'No internet service' and 'No phone service' to 'No' for counting
    df_services = df[service_cols].replace(['No internet service', 'No phone service'], 'No')
    df['services_count'] = (df_services == 'Yes').sum(axis=1)

    # no_tech_support_flag
    df['no_tech_support_flag'] = ((df['InternetService'] != 'No') & (df['TechSupport'] == 'No')).astype(int)

    # CLV Proxy Calculation
    # Assumption: CLV is proxied by historical revenue.
    df['CLV'] = df['MonthlyCharges'] * df['tenure']

    return df


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Creates a ColumnTransformer for preprocessing."""
    # Identify categorical and numerical features
    # Exclude CLV from scaling as it's an outcome, not a predictor for the model
    numerical_features = X.select_dtypes(include=np.number).columns.drop(['CLV'])
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Create preprocessing pipelines for both feature types
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep CLV column as is
    )
    return preprocessor


def run_data_prep():
    """Main function to orchestrate the data preparation process."""
    print("--- Starting Data Preparation ---")
    create_directories()

    # 1. Load and Clean Data
    df = load_data(DATA_URL, RAW_DATA_PATH)
    df_cleaned = clean_data(df.copy())

    # 2. Engineer Features
    df_featured = engineer_features(df_cleaned)

    # 3. Define Features (X) and Target (y)
    # Drop customerID as it's an identifier, not a feature
    X = df_featured.drop(columns=['Churn', 'customerID'])
    y = df_featured['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    # 4. Split Data (60/20/20 train/val/test) with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"Data split into: Train ({X_train.shape[0]}), Val ({X_val.shape[0]}), Test ({X_test.shape[0]})")

    # 5. Create and Fit Preprocessing Pipeline
    preprocessor = create_preprocessor(X_train)
    preprocessor.fit(X_train)

    # 6. Save the fitted preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    # 7. Apply preprocessing and save data splits
    # We save the splits *before* processing to keep them human-readable
    # The preprocessor will be applied on-the-fly during training
    X_train.join(y_train).to_csv(os.path.join(PROCESSED_DATA_PATH, "train.csv"), index=False)
    X_val.join(y_val).to_csv(os.path.join(PROCESSED_DATA_PATH, "val.csv"), index=False)
    X_test.join(y_test).to_csv(os.path.join(PROCESSED_DATA_PATH, "test.csv"), index=False)
    print(f"Processed data splits saved to {PROCESSED_DATA_PATH}")

    print("--- Data Preparation Complete ---")


if __name__ == "__main__":
    run_data_prep()