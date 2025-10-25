import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve, average_precision_score
import base64
import re
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn & CLV Prediction",
    page_icon="🚀",
    layout="wide"
)

# --- Path Configuration ---
MODELS_DIR = "models/"
DATA_DIR = "data/processed/"
ASSETS_DIR = "assets/"
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")

# --- Caching ---
@st.cache_data
def load_data():
    """Loads performance metrics and data for CLV analysis."""
    try:
        performance_df = pd.read_csv(os.path.join(DATA_DIR, "model_performance.csv"), index_col=0)
        # Load the test data to use for CLV analysis visualizations
        test_df_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
        # Load train data to get column order for prediction
        train_df_raw = pd.read_csv(TRAIN_DATA_PATH)
        return performance_df, test_df_raw, train_df_raw
    except FileNotFoundError:
        st.error(f"Data files not found. Please run `src/data_prep.py` and `src/train_models.py` first.")
        return None, None, None

@st.cache_resource
def load_models():
    """Loads all trained model pipelines."""
    models = {}
    try:
        models["LogisticRegression"] = joblib.load(os.path.join(MODELS_DIR, "logisticregression.pkl"))
        models["RandomForest"] = joblib.load(os.path.join(MODELS_DIR, "randomforest.pkl"))
        models["XGBoost"] = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
        return models
    except FileNotFoundError:
        st.error(f"Model files not found. Please run `src/train_models.py` first.")
        return None

@st.cache_resource
def get_tree_explainer(_model_pipeline):
    """Creates and caches a SHAP TreeExplainer for a given model pipeline."""
    # SHAP explainers work on the final model, not the whole pipeline.
    # We extract the model step ('classifier') from the pipeline.
    model = _model_pipeline.named_steps['classifier']
    
    # The preprocessor is the first step in the pipeline
    preprocessor = _model_pipeline.named_steps['preprocessor']
    
    # SHAP needs the background data to be preprocessed
    # We'll use a sample of the training data for this
    train_df_sample = train_df.drop('Churn', axis=1).sample(100, random_state=42)
    train_df_processed = preprocessor.transform(train_df_sample)
    
    # Create the explainer on the model and the processed background data
    return shap.TreeExplainer(model, train_df_processed)

def create_input_df(user_inputs: dict, train_cols: list) -> pd.DataFrame:
    """
    Creates a DataFrame from user inputs that matches the training data structure.
    """
    # Start with a dictionary of all zeros
    data = {col: [0] for col in train_cols}
    
    # Update with user inputs
    for key, value in user_inputs.items():
        data[key] = [value]

    # Create DataFrame
    input_df = pd.DataFrame(data)

    # Re-engineer features that were created in data_prep.py
    bins = [-1, 6, 12, 24, 73]
    labels = ['0-6m', '7-12m', '13-24m', '24m+']
    input_df['tenure_bucket'] = pd.cut(input_df['tenure'], bins=bins, labels=labels)

    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df_services = input_df[service_cols].replace(['No internet service', 'No phone service'], 'No')
    input_df['services_count'] = (df_services == 'Yes').sum(axis=1)
    input_df['no_tech_support_flag'] = ((input_df['InternetService'] != 'No') & (input_df['TechSupport'] == 'No')).astype(int)
    input_df['CLV'] = input_df['MonthlyCharges'] * input_df['tenure']

    # Ensure column order is the same as training
    return input_df[train_cols]

def get_image_as_base64(path):
    """Encodes an image file to a base64 string for embedding in HTML."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def clean_feature_names(feature_names: list) -> list:
    """Cleans up feature names from the preprocessor for better display."""
    cleaned_names = []
    for name in feature_names:
        # Remove prefixes added by ColumnTransformer
        name = re.sub(r'^(num__|cat__|remainder__)', '', name)
        # Replace underscores with a colon and space for categorical features
        name = name.replace('_', ': ')
        # Add spaces before uppercase letters in camelCase names
        name = re.sub(r'(?<!^)(?=[A-Z])', ' ', name)
        cleaned_names.append(name)
    return cleaned_names


# --- Load Resources ---
performance_df, test_df, train_df = load_data()
models = load_models()

# --- App Title ---
st.title("SaaS Customer Churn Prediction & CLV Analysis")

# --- Author Header ---
profile_pic_path = os.path.join(ASSETS_DIR, "profile.png")
b64_image = get_image_as_base64(profile_pic_path)

if b64_image:
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{b64_image}" style="border-radius: 50%; width: 90px; height: 90px;">
            <div style="line-height: 1.2;">
                <div style="font-size: 0.9rem; color: #888; margin: 0;">Designed by</div>
                <div style="font-size: 1.75rem; font-weight: bold; margin: 0; padding: 0;">Raphael Daveal</div>
                <div style="font-size: 1rem; color: #555; margin: 0;">Machine Learning Engineer</div>
                <div style="margin-top: 4px;">
                    <a href="https://www.linkedin.com/in/daveralphy/" target="_blank" style="text-decoration: none;"><i>LinkedIn</i></a> | 
                    <a href="https://github.com/daveralphy" target="_blank" style="text-decoration: none;"><i>GitHub</i></a>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.subheader("Application Overview")
st.markdown("""
This application predicts customer churn and analyzes Customer Lifetime Value (CLV) to help prioritize retention efforts.
Navigate through the tabs to predict churn for a single customer, review model performance, or explore the CLV analysis.
""")

# --- Main Application ---
if all(v is not None for v in [performance_df, test_df, train_df, models]):
    tabs = st.tabs(["🚀 Predict Churn", "📊 Model Performance", "💰 CLV Overview"])

    # --- TAB 1: Predict Churn ---
    with tabs[0]:
        st.header("Predict a Single Customer's Churn Risk")
        
        with st.form("prediction_form"):
            # Get the list of columns the model was trained on, excluding the target
            X_train_cols = train_df.drop(columns=['Churn']).columns.tolist()

            # Create three columns for the input sections
            c1, c2, c3 = st.columns(3, gap="large")

            with c1:
                with st.container(border=True):
                    st.subheader("👤 Customer Profile")
                    gender = st.selectbox("Gender", options=train_df['gender'].unique(), index=0)
                    senior_citizen = st.radio("Senior Citizen?", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
                    partner = st.radio("Has a Partner?", options=['Yes', 'No'], horizontal=True)
                    dependents = st.radio("Has Dependents?", options=['Yes', 'No'], horizontal=True)

            with c2:
                with st.container(border=True):
                    st.subheader("📄 Account & Billing")
                    tenure = st.slider("Tenure (months)", 0, 72, 24)
                    contract = st.selectbox("Contract Type", options=train_df['Contract'].unique())
                    paperless_billing = st.radio("Paperless Billing?", options=['Yes', 'No'], horizontal=True)
                    payment_method = st.selectbox("Payment Method", options=train_df['PaymentMethod'].unique())
                    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 75.0, 0.01)
                    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))

            with c3:
                with st.container(border=True):
                    st.subheader("🔧 Subscribed Services")
                    phone_service = st.selectbox("Phone Service", options=train_df['PhoneService'].unique())
                    multiple_lines = st.selectbox("Multiple Lines", options=train_df['MultipleLines'].unique())
                    internet_service = st.selectbox("Internet Service", options=train_df['InternetService'].unique())
                    online_security = st.selectbox("Online Security", options=train_df['OnlineSecurity'].unique())
                    online_backup = st.selectbox("Online Backup", options=train_df['OnlineBackup'].unique())
                    device_protection = st.selectbox("Device Protection", options=train_df['DeviceProtection'].unique())
                    tech_support = st.selectbox("Tech Support", options=train_df['TechSupport'].unique())
                    streaming_tv = st.selectbox("Streaming TV", options=train_df['StreamingTV'].unique())
                    streaming_movies = st.selectbox("Streaming Movies", options=train_df['StreamingMovies'].unique())

            submit_button = st.form_submit_button(label="Predict Churn Risk", type="primary")

        # When the button is clicked...
        if submit_button:
            user_inputs = {
                'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner, 'Dependents': dependents,
                'tenure': tenure, 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
                'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
                'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
            }
            
            input_df = create_input_df(user_inputs, X_train_cols)
            
            # Use XGBoost as it's typically the best performer
            model_pipeline = models["XGBoost"]
            churn_proba = model_pipeline.predict_proba(input_df)[0][1]
            
            # Define risk level
            if churn_proba > 0.6:
                risk_level = "High"
                delta_color = "inverse"
            elif churn_proba > 0.3:
                risk_level = "Medium"
                delta_color = "normal"
            else:
                risk_level = "Low"
                delta_color = "off"

            st.divider()
            res_col1, res_col2 = st.columns([1, 2], gap="large")
            with res_col1:
                st.subheader("Prediction Results")
                st.metric("Churn Probability", f"{churn_proba:.1%}", f"{risk_level} Risk", delta_color=delta_color)
                
                st.subheader("Estimated CLV")
                clv = monthly_charges * tenure
                st.metric("Estimated CLV (Historical)", f"${clv:,.2f}")
                st.code(f"CLV = MonthlyCharges × Tenure\nCLV = ${monthly_charges:.2f} × {tenure} months", language="text")

            with res_col2:
                st.subheader("Prediction Explanation (SHAP)")
                with st.spinner("Generating explanation..."):
                    explainer = get_tree_explainer(model_pipeline)
                    
                    # We need to preprocess the single input row for the explainer
                    preprocessor = model_pipeline.named_steps['preprocessor']
                    input_processed = preprocessor.transform(input_df)
                    
                    shap_values = explainer.shap_values(input_processed)
                    
                    # Create the waterfall plot
                    fig, ax = plt.subplots()
                    ax.set_facecolor("white")
                    fig.set_facecolor("white")
                    plt.rcParams['text.color'] = '#31333F'
                    
                    processed_feature_names = preprocessor.get_feature_names_out()
                    shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                                         base_values=explainer.expected_value, 
                                                         data=input_processed[0],
                                                         feature_names=clean_feature_names(processed_feature_names)),
                                        show=False)
                    plt.tight_layout()
                    ax.set_xlabel("") # Remove the SHAP x-axis label
                    st.pyplot(fig, use_container_width=True)

    # --- TAB 2: Model Performance ---
    with tabs[1]:
        st.header("Model Performance Comparison")
        st.dataframe(performance_df.style.format("{:.4f}").background_gradient(cmap='viridis', subset=['AUC-ROC', 'F1-Score']))
        
        with st.expander("Metric Definitions"):
            st.markdown("""
            *   **Precision**: Of all customers we predicted to churn, what percentage actually churned? (Focus on avoiding false positives).
            *   **Recall**: Of all customers who actually churned, what percentage did we correctly identify? (Focus on finding all churners).
            *   **AUC-ROC**: Measures the model's ability to distinguish between positive and negative classes across all thresholds. 1.0 is perfect.
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("ROC Curves")
                fig = go.Figure()
                fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

                for model_name, model in models.items():
                    y_proba = model.predict_proba(test_df.drop('Churn', axis=1))[:, 1]
                    fpr, tpr, _ = roc_curve(test_df['Churn'], y_proba)
                    auc = performance_df.loc[model_name, 'AUC-ROC']
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{model_name} (AUC={auc:.4f})", mode='lines'))

                fig.update_layout(
                    title="ROC Curves for All Models",
                    xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
                    legend_title="Model", template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            with st.container(border=True):
                st.subheader("Confusion Matrix")
                model_to_show = st.selectbox("Select Model for Confusion Matrix", list(models.keys()), key="cm_select")
                y_pred = models[model_to_show].predict(test_df.drop('Churn', axis=1))
                cm = confusion_matrix(test_df['Churn'], y_pred)
                fig = px.imshow(cm, text_auto=True,
                                labels=dict(x="Predicted", y="Actual", color="Count"),
                                x=['Not Churn', 'Churn'], y=['Not Churn', 'Churn'],
                                color_continuous_scale=px.colors.sequential.Purples,
                                title=f"Confusion Matrix: {model_to_show}")
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        col3, col4 = st.columns(2)

        with col3:
            with st.container(border=True):
                st.subheader("Precision-Recall Curves")
                st.write("Focuses on the performance of predicting the minority class (Churn = Yes).")
                fig = go.Figure()
                for model_name, model in models.items():
                    y_proba = model.predict_proba(test_df.drop('Churn', axis=1))[:, 1]
                    precision, recall, _ = precision_recall_curve(test_df['Churn'], y_proba)
                    ap = average_precision_score(test_df['Churn'], y_proba)
                    fig.add_trace(go.Scatter(x=recall, y=precision, name=f"{model_name} (AP={ap:.3f})", mode='lines'))

                fig.update_layout(title="Precision-Recall Curves for All Models",
                                  xaxis_title='Recall', yaxis_title='Precision',
                                  legend_title="Model", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            with st.container(border=True):
                st.subheader("Global Feature Importance (SHAP)")
                st.write("Top features influencing the XGBoost model's predictions across the test set.")
                with st.spinner("Generating SHAP summary..."):
                    model_pipeline = models["XGBoost"]
                    explainer = get_tree_explainer(model_pipeline)
                    preprocessor = model_pipeline.named_steps['preprocessor']
                    
                    test_sample = test_df.drop('Churn', axis=1).sample(200, random_state=42)
                    test_sample_processed = preprocessor.transform(test_sample)
                    
                    shap_values_summary = explainer.shap_values(test_sample_processed)
                    
                    fig, ax = plt.subplots()
                    ax.set_facecolor("white")
                    fig.set_facecolor("white")
                    plt.rcParams['text.color'] = '#31333F'
                    processed_feature_names = preprocessor.get_feature_names_out()
                    shap.summary_plot(shap_values_summary, test_sample_processed, 
                                      feature_names=clean_feature_names(processed_feature_names), 
                                      show=False, plot_type='bar', color_bar=False, plot_size=None)
                    ax.set_xlabel("") # Remove the SHAP x-axis label
                    ax.grid(zorder=0) # Add grid behind the bars
                    st.pyplot(fig, use_container_width=True)

    # --- TAB 3: CLV Overview ---
    with tabs[2]:
        st.header("Customer Lifetime Value (CLV) Analysis")
        st.markdown("This section analyzes the relationship between the historical CLV proxy and customer churn based on the test dataset.")

        # --- IMPORTANT: Work on a copy to avoid caching issues ---
        analysis_df = test_df.copy()

        # Define segment labels and create the segments using quartiles
        segment_labels = ['Low', 'Medium', 'High', 'Premium']
        try:
            analysis_df['clv_segment'] = pd.qcut(analysis_df['CLV'], 4, labels=segment_labels, duplicates='drop')
        except ValueError as e:
            st.error(f"Could not create CLV segments. The data distribution may be too skewed. Error: {e}")
            st.stop()

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("CLV Distribution")
            fig_dist = px.histogram(analysis_df, x="CLV", nbins=50, title="Distribution of Customer Lifetime Value (CLV)",
                                   color_discrete_sequence=['#6366F1'])
            fig_dist.update_layout(xaxis_title="CLV ($)", yaxis_title="Number of Customers", template="plotly_white")
            st.plotly_chart(fig_dist, use_container_width=True)

        with col2:
            st.subheader("Churn Rate by CLV Quartile")
            churn_by_segment = analysis_df.groupby('clv_segment', observed=False)['Churn'].mean().reset_index()
            fig_churn = px.bar(churn_by_segment, x='clv_segment', y='Churn',
                         labels={'Churn': 'Churn Rate', 'clv_quartile': 'CLV Quartile'}, text_auto='.1%',
                         color='clv_segment', category_orders={"clv_segment": segment_labels},
                         color_discrete_map={'Low': '#A5B4FC', 'Medium': '#818CF8', 'High': '#6366F1', 'Premium': '#4F46E5'})
            fig_churn.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_churn, use_container_width=True)

        st.divider()
        st.subheader("Key Business Takeaway")
        # --- Dynamic Insight Generation ---
        churn_by_segment = analysis_df.groupby('clv_segment', observed=False)['Churn'].mean().reset_index()
        highest_churn_segment = churn_by_segment.sort_values('Churn', ascending=False).iloc[0]
        lowest_churn_segment = churn_by_segment.sort_values('Churn', ascending=True).iloc[0]
        average_churn_rate = analysis_df['Churn'].mean()
        
        st.success(f"""
        **Insight**: The **{highest_churn_segment['clv_segment']}** customer quartile is the most vulnerable, with a churn rate of **{highest_churn_segment['Churn']:.1%}**. 
        This is significantly higher than the average churn rate of {average_churn_rate:.1%} and starkly contrasts with the **{lowest_churn_segment['clv_segment']}** quartile's rate of just {lowest_churn_segment['Churn']:.1%}.
        
        **Recommendation**: Immediately prioritize retention campaigns for the **High** and **Premium** CLV customers. 
        Given their high historical value, the ROI on retaining them is substantial. A targeted campaign offering proactive support, loyalty discounts, or a contract review could be highly effective.
        """)

else:
    st.warning("Could not load necessary data or model files. Please ensure you have run the setup scripts.")