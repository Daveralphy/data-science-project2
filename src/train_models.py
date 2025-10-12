import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Configuration ---
PROCESSED_DATA_PATH = "data/processed/"
MODELS_DIR = "models/"
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")


def load_data(split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Loads a data split from the processed folder."""
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, f"{split_name}.csv"))
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y


def evaluate_model(model_name: str, y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> dict:
    """Calculates evaluation metrics for a given model."""
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-ROC": roc_auc_score(y_true, y_proba)
    }
    print(f"Evaluation for {model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    return metrics


def run_training():
    """
    Loads processed data, trains models, evaluates, and saves them as pipelines.
    """
    print("--- Starting Model Training ---")

    # 1. Load Data
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")
    X_test, y_test = load_data("test")

    # For final model training, combine train and validation sets as per project spec
    X_train_full = pd.concat([X_train, X_val], ignore_index=True)
    y_train_full = pd.concat([y_train, y_val], ignore_index=True)

    # 2. Load the preprocessor
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("Preprocessor loaded successfully.")

    # 3. Define Models
    # Handle class imbalance using class_weight or scale_pos_weight
    imbalance_ratio = y_train_full.value_counts()[0] / y_train_full.value_counts()[1]

    models = {
        "LogisticRegression": LogisticRegression(
            solver='liblinear', class_weight='balanced', random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            objective='binary:logistic', eval_metric='logloss',
            scale_pos_weight=imbalance_ratio, use_label_encoder=False,
            max_depth=5, learning_rate=0.1, n_estimators=200, random_state=42
        )
    }

    # 4. Train, Evaluate, and Save Models
    performance_metrics = {}
    for name, model in models.items():
        print(f"\n--- Training {name} ---")

        # Create a full pipeline including the preprocessor and the model
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Train the pipeline on the full training data
        pipeline.fit(X_train_full, y_train_full)

        # Make predictions on the test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Evaluate the model
        performance_metrics[name] = evaluate_model(name, y_test, y_pred, y_proba)

        # Save the entire pipeline
        model_path = os.path.join(MODELS_DIR, f"{name.lower()}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Saved {name} pipeline to {model_path}")

    # 5. Save performance metrics for the app
    performance_df = pd.DataFrame(performance_metrics).T
    performance_df.to_csv(os.path.join(PROCESSED_DATA_PATH, "model_performance.csv"))
    print(f"\nModel performance metrics saved to {os.path.join(PROCESSED_DATA_PATH, 'model_performance.csv')}")

    print("--- Model Training Complete ---")


if __name__ == "__main__":
    run_training()