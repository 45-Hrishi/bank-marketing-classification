
import os
import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Marketing Subscription Prediction")
st.write(
    "Compare five tuned machine learning classifiers "
    "on an uploaded test dataset."
)

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

MODEL_FILES = {
    "Logistic Regression":
        "models/logistic_regression_tuned_pipeline.joblib",

    "Decision Tree":
        "models/decision_tree_tuned_pipeline.joblib",

    "K-Nearest Neighbors":
        "models/knn_tuned_pipeline.joblib",

    "Gaussian Naive Bayes":
        "models/gaussian_naive_bayes_tuned_pipeline.joblib",

    "Support Vector Machine":
        "models/svm_tuned_pipeline.joblib"
}


@st.cache_resource
def load_models():
    loaded = {}

    for name, path in MODEL_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}"
            )

        loaded[name] = joblib.load(path)

    return loaded


# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------

try:
    models = load_models()
    st.success("All five tuned models loaded successfully.")

except Exception as error:
    st.error(f"Unable to load models: {error}")
    st.stop()


# --------------------------------------------------
# UPLOAD TEST DATA
# --------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file containing the 15 input features and target column y",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_df = pd.read_csv(uploaded_file)

        if "y" not in test_df.columns:
            st.error(
                "The uploaded CSV must contain the actual target column 'y'."
            )
            st.stop()

        if test_df.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()

        # Separate actual target and input features
        X_test = test_df.drop(columns=["y"])
        y_test = test_df["y"]

        # Support both yes/no and 0/1 target formats
        if y_test.isin(["yes", "no"]).all():
            y_test = y_test.map({"no": 0, "yes": 1})

        elif y_test.isin([0, 1]).all():
            y_test = y_test.astype(int)

        else:
            st.error(
                "Target column 'y' must contain only yes/no or 0/1."
            )
            st.stop()

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(test_df.head(10), use_container_width=True)

        st.write("Number of test records:", len(test_df))
        st.write("Number of input features:", X_test.shape[1])

        # --------------------------------------------------
        # VALIDATE FEATURE COLUMNS
        # --------------------------------------------------

        reference_model = next(iter(models.values()))
        expected_columns = list(reference_model.feature_names_in_)

        missing_columns = [
            col for col in expected_columns
            if col not in X_test.columns
        ]

        extra_columns = [
            col for col in X_test.columns
            if col not in expected_columns
        ]

        if missing_columns or extra_columns:
            st.error("Uploaded dataset columns do not match training data.")

            if missing_columns:
                st.write("Missing columns:", missing_columns)

            if extra_columns:
                st.write("Unexpected columns:", extra_columns)

            st.stop()

        # Keep columns in training order
        X_test = X_test[expected_columns]

        # --------------------------------------------------
        # EVALUATE ALL MODELS
        # --------------------------------------------------

        st.header("2. Model Evaluation")

        all_results = {}
        metrics_list = []

        for model_name, model in models.items():

            predictions = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                scores = model.predict_proba(X_test)[:, 1]
            else:
                scores = model.decision_function(X_test)

            metrics = {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "AUC": roc_auc_score(y_test, scores),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "F1 Score": f1_score(
                    y_test, predictions, zero_division=0
                ),
                "MCC": matthews_corrcoef(y_test, predictions)
            }

            metrics_list.append(metrics)

            all_results[model_name] = {
                "predictions": predictions,
                "metrics": metrics,
                "scores": scores
            }

        results_df = pd.DataFrame(metrics_list)

        st.subheader("Comparison of All Five Models")
        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.4f}",
                "AUC": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1 Score": "{:.4f}",
                "MCC": "{:.4f}"
            }),
            use_container_width=True
        )

        # Download evaluation results
        st.download_button(
            "Download Model Comparison CSV",
            results_df.to_csv(index=False),
            file_name="streamlit_model_comparison.csv",
            mime="text/csv"
        )

        # --------------------------------------------------
        # SELECT MODEL
        # --------------------------------------------------

        st.header("3. Detailed Model Results")

        selected_model = st.selectbox(
            "Select a model",
            list(models.keys())
        )

        selected_result = all_results[selected_model]

        selected_metrics = selected_result["metrics"]
        predictions = selected_result["predictions"]

        metric_columns = st.columns(3)

        for index, metric_name in enumerate([
            "Accuracy", "AUC", "Precision",
            "Recall", "F1 Score", "MCC"
        ]):

            metric_columns[index % 3].metric(
                metric_name,
                f"{selected_metrics[metric_name]:.4f}"
            )

        # --------------------------------------------------
        # CONFUSION MATRIX
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")
        ax.set_title(f"{selected_model} - Confusion Matrix")

        st.pyplot(fig)
        plt.close(fig)

        # --------------------------------------------------
        # CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader("Classification Report")

        report = classification_report(
            y_test,
            predictions,
            target_names=["No", "Yes"],
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).transpose(),
            use_container_width=True
        )

        # --------------------------------------------------
        # PREDICTION RESULTS
        # --------------------------------------------------

        st.subheader("Prediction Results")

        prediction_df = X_test.copy()

        prediction_df["Actual"] = y_test.values
        prediction_df["Predicted"] = predictions
        prediction_df["Actual Label"] = prediction_df["Actual"].map(
            {0: "No", 1: "Yes"}
        )
        prediction_df["Predicted Label"] = prediction_df["Predicted"].map(
            {0: "No", 1: "Yes"}
        )

        st.dataframe(
            prediction_df.head(50),
            use_container_width=True
        )

        st.download_button(
            "Download Prediction Results",
            prediction_df.to_csv(index=False),
            file_name="bank_marketing_predictions.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Error while evaluating uploaded data: {error}")

else:
    st.info("Upload test_data.csv to begin model evaluation.")

st.sidebar.title("About This Project")
st.sidebar.write(
    "Bank Marketing Classification Project"
)
st.sidebar.write("Five tuned machine learning algorithms")
st.sidebar.write("GridSearchCV with 5-fold Stratified CV")
st.sidebar.write("Six classification evaluation metrics")
st.sidebar.caption("Models are loaded from saved joblib pipelines.")
