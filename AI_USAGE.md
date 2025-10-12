# AI Usage Documentation

This document outlines how I used AI assistance, specifically **Google's Gemini Code Assist**, in the development of the "SaaS Customer Churn & Lifetime Value Prediction" project. The goal of writing this is to maintain transparency about my development process.

---

### 1. Project Scaffolding and Best Practices

I used AI assistance to help set up the initial project structure and establish coding best practices.

*   **Directory Structure:** I received guidance on creating a logical folder structure (`data/`, `src/`, `models/`, `assets/`).
*   **Environment Setup:** I used Gemini to generate the initial `.gitignore` file to exclude virtual environments and Python cache files from version control.
*   **Scripting Foundation:** I also leveraged the AI to provide boilerplate code for the `data_prep.py` and `train_models.py` scripts, including function definitions and the use of `os.path.join` for platform-agnostic path handling.

### 2. Streamlit UI Development and Refinement

A significant portion of my work with the AI focused on building and iteratively refining the user interface in `app.py`.

*   **Layout and Structure:** I used the AI to help create the main tabbed layout (`st.tabs`) and organize the prediction form into columns (`st.columns`) with containers (`st.container(border=True)`).
*   **Custom Header Design:** This was a multi-step, iterative process:
    *   Generate the initial two-column layout for my profile picture and bio.
    *   Provide custom CSS to make the profile image circular.
    *   Refactor the text bio into a single `st.markdown` block with custom HTML and inline CSS for precise control over line spacing.
    *   Develop the final header implementation using a `flexbox` layout and a base64-encoded image for a professional author byline.
*   **Dynamic UI Elements:** I then wrote the logic, with AI help, to display the churn prediction result using `st.metric` with dynamic coloring based on the predicted risk level.

### 3. Data Visualization and Model Interpretability (SHAP)

I used AI to help implement the advanced visualization and model explanation features.

*   **Plotly Charts:** I had the AI generate the code for creating the ROC curves, Precision-Recall curves, and CLV analysis charts using `plotly.express` and `plotly.graph_objects`.
*   **SHAP Integration:**
    *   I used Gemini to write the `get_tree_explainer` function to correctly instantiate a `shap.TreeExplainer` on the model (not the full pipeline) and a background dataset.
    *   I then generated the code to create and display the SHAP waterfall plot for single-customer predictions and the summary bar plot for global feature importance.
    *   I also used it to create the `clean_feature_names` helper function with regular expressions to format the feature names from the `scikit-learn` pipeline for better readability on the SHAP plots.

### 4. Code Refactoring and Performance Optimization

*   **Caching:** Following AI recommendations, I implemented `@st.cache_data` and `@st.cache_resource` decorators to prevent redundant loading of data and models, which significantly improved the application's performance.
*   **Error Handling:** I added `try...except` blocks to the data and model loading functions to provide user-friendly error messages if essential files are missing.
*   **Dynamic Insights:** I wrote the Python code, with AI assistance, to dynamically generate the "Key Business Takeaway" on the CLV tab by analyzing the results of the `churn_by_clv` DataFrame.

### 5. Documentation and Deployment

*   **README Generation:** I used the AI to help create and refine the `README.md` file, including adding the live application URL, embedding the screenshot, and structuring the "How to Run Locally" instructions.
*   **Deployment Guidance:** I followed a step-by-step guide provided by the AI to deploy the application to Streamlit Community Cloud, which included finalizing the `requirements.txt` file.
*   **AI Usage File:** Finally, I used Gemini to generate the initial draft of this document to summarize my use of AI in the project.