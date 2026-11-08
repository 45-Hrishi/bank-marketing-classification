# Bank Marketing Subscription Prediction Using Machine Learning

## 1. Project Overview

This project develops and evaluates machine learning classification
models to predict whether a bank client will subscribe to a term deposit
based on customer and marketing campaign attributes. It uses the UCI
Bank Marketing dataset and implements five classifiers, hyperparameter
tuning, test-set evaluation, and an interactive Streamlit dashboard.

The workflow covers dataset preparation, exploratory data analysis,
preprocessing, model training, model comparison, artifact saving, and
deployment.

## 2. Project Information

| Item | Details |
|---|---|
| Programme | M.Tech – Artificial Intelligence and Machine Learning |
| Course | Machine Learning |
| Assignment | Assignment 2 |
| Problem Type | Binary Classification |
| Dataset | UCI Bank Marketing (`bank-full.csv`) |
| Development Environment | BITS Virtual Lab |
| Application Framework | Streamlit |
| GitHub Repository | [(https://github.com/45-Hrishi/bank-marketing-classification.git)](GITHUB_URL) |
| Live Application | [https://bank-marketing-classification-bitsp-aiml.streamlit.app/](STREAMLIT_URL) |

## 3. Problem Statement

Bank marketing campaigns contact customers to promote term deposits.
Contacting every customer may consume substantial time and resources.
The objective of this project is to build classification models that
predict whether a customer is likely to subscribe to a term deposit,
using the customer and campaign information available in the dataset.

The target variable is `y`: - `yes` (encoded as `1`): the customer
subscribed. - `no` (encoded as `0`): the customer did not subscribe.

The `duration` feature is excluded from model training because it
records the duration of the last contact. In a pre-call prediction
setting, this value is not known before the call and could introduce
target leakage.

## 4. Dataset Description

**Source:** UCI Machine Learning Repository -- [Bank Marketing
Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

The project uses the `bank-full.csv` file. The original dataset contains
45,211 records, 16 input attributes, and the target attribute `y`. It
includes demographic, financial, contact, and campaign-related
information.

The original input attributes include:

 | Feature | Description |
|---|---|
| `age` | Customer age |
| `job` | Type of job |
| `marital` | Marital status |
| `education` | Education level |
| `default` | Whether the customer has credit in default |
| `balance` | Average yearly account balance |
| `housing` | Whether the customer has a housing loan |
| `loan` | Whether the customer has a personal loan |
| `contact` | Contact communication type |
| `day` | Last contact day of the month |
| `month` | Last contact month |
| `duration` | Duration of the last contact |
| `campaign` | Number of contacts during the current campaign |
| `pdays` | Days since the customer was last contacted in a previous campaign |
| `previous` | Number of contacts before the current campaign |
| `poutcome` | Outcome of the previous marketing campaign |
| `y` | Target: term deposit subscription (`yes`/`no`) |
  -----------------------------------------------------------------------

After excluding `duration`, the modelling dataset uses **15 input
features and one target variable**. The original dataset exceeds the
assignment minimum of 12 features and 500 instances.

## 5. Data Preparation and Exploratory Analysis

The data preparation workflow includes: - Loading the
semicolon-separated `bank-full.csv` dataset. - Inspecting dataset
dimensions, columns, data types, and summary statistics. - Checking
missing values and duplicate records. - Removing duplicate records where
identified. - Exploring the target-class distribution and feature
distributions. - Encoding the target as `yes = 1` and `no = 0`. -
Excluding `duration` to avoid using information unavailable at the
intended prediction time. - Separating numerical and categorical input
features.

### Preprocessing used in model pipelines

Numerical features are imputed using the median and standardised using
`StandardScaler`. Categorical features are imputed using the most
frequent value and transformed using one-hot encoding with
unknown-category handling.

Preprocessing is fitted on the training data within each model pipeline.
This helps prevent information from the test set leaking into model
fitting.

## 6. Machine Learning Models

All five required classification algorithms are implemented on the same
dataset:

1.  **Logistic Regression** -- a linear probabilistic classifier used as
    a baseline.
2.  **Decision Tree Classifier** -- a tree-based classifier that learns
    feature-based decision rules.
3.  **K-Nearest Neighbors (KNN)** -- a distance-based classifier that
    predicts using nearby training observations.
4.  **Gaussian Naive Bayes** -- a probabilistic classifier based on
    Bayes' theorem and Gaussian likelihoods for numerical inputs.
5.  **Support Vector Machine (SVM)** -- a margin-based classifier
    implemented using `LinearSVC`.

## 7. Experimental Setup and Hyperparameter Tuning

The dataset is split into training and test subsets using an 80:20
split, with stratification and `random_state=42`. Hyperparameter
selection is performed using `GridSearchCV` with five-fold stratified
cross-validation. The search objective is the F1-score.

The following are the hyperparameter search spaces used in the
experiment:

| Model | Hyperparameters Explored |
|---|---|
| Logistic Regression | `C`: [0.1, 1, 10]; `solver`: [`lbfgs`, `liblinear`] |
| Decision Tree | `max_depth`: [8, 12, 16]; `min_samples_split`: [10, 20]; `min_samples_leaf`: [5, 10]; `criterion`: [`gini`, `entropy`] |
| KNN | `n_neighbors`: [5, 7, 11]; `weights`: [`uniform`, `distance`]; `p`: [1, 2] |
| Gaussian Naive Bayes | `var_smoothing`: [1e-9, 1e-8, 1e-7] |
| Linear SVM | `C`: [0.01, 0.1, 1]; `loss`: [`hinge`, `squared_hinge`] |
  -----------------------------------------------------------------------

**Common tuning configuration**

| Setting | Value |
|---|---|
| Search Method | `GridSearchCV` |
| Cross-validation | `StratifiedKFold`, 5 folds |
| Shuffle | `True` |
| Cross-validation Random State | `42` |
| Search Scoring | `f1` |
| Train/Test Split | 80% / 20% |
| Split Random State | `42` |
| Split Stratification | Target variable |

The parameter ranges above describe the values explored during tuning;
they are not the selected best parameters. The exact `best_params_`
returned by the fitted searches can be added to this README if desired.

## 8. Evaluation Metrics

Each tuned model is evaluated on the held-out test set using the six
metrics required by the assignment:

| Metric | Description |
|---|---|
| **Accuracy** | Proportion of all predictions that are correct. |
| **AUC** | Area under the ROC curve; measures the model's ability to rank positive cases above negative cases. |
| **Precision** | Proportion of predicted positive cases that are actually positive. |
| **Recall** | Proportion of actual positive cases correctly identified. |
| **F1-score** | Harmonic mean of precision and recall. |
| **Matthews Correlation Coefficient (MCC)** | Correlation between actual and predicted classes, taking all four confusion-matrix outcomes into account. |

AUC for a model without `predict_proba` can be calculated from its
decision scores. The reported values below are test-set results.

## 9. Tuned Model Performance Comparison

The following table reports the evaluation results for the five tuned
models on the held-out test dataset.

| ML Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7554 | **0.7722** | 0.2666 | **0.6229** | 0.3734 | 0.2855 |
| Decision Tree | 0.8252 | 0.7661 | 0.3477 | 0.5643 | **0.4303** | **0.3475** |
| KNN | **0.8874** | 0.7119 | **0.5444** | 0.2316 | 0.3249 | 0.3043 |
| Gaussian Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine (LinearSVC) | 0.7917 | 0.7650 | 0.2953 | 0.5633 | 0.3875 | 0.2974 |                                                                
  -------------------------------------------------------------------------------------------

*Values are rounded to four decimal places. Bold values indicate the
highest value in each metric column.*

## 10. Model-wise Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 0.7554 and the highest AUC
of 0.7722 among the evaluated models. Its recall of 0.6229 was also the
highest, indicating that it identified the largest proportion of actual
subscribers. However, its precision of 0.2666 was low, meaning that many
customers predicted as subscribers were not actual subscribers.

### Decision Tree

The Decision Tree achieved an accuracy of 0.8252, an F1-score of 0.4303,
and an MCC of 0.3475. It obtained the highest F1-score and MCC among the
five models, indicating the strongest balance of precision and recall
and the strongest overall correlation between predictions and actual
labels according to MCC. Its recall of 0.5643 was also relatively high.

### K-Nearest Neighbors (KNN)

KNN achieved the highest accuracy (0.8874) and precision (0.5444).
However, its recall was 0.2316, the lowest among the five models. This
means it missed a substantial proportion of actual subscribers.
Therefore, its high accuracy alone does not make it the best choice when
identifying subscribers is important.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 0.8452, an F1-score of
0.4123, and an MCC of 0.3271. Its precision (0.3708) and recall (0.4641)
were more balanced than those of some other models. It was competitive,
although its F1-score and MCC were below the Decision Tree results.

### Support Vector Machine (LinearSVC)

The Linear SVM achieved an accuracy of 0.7917, an AUC of 0.7650, and a
recall of 0.5633. Its recall was close to that of the Decision Tree, but
its precision and F1-score were lower. It provides a reasonable level of
positive-class identification but did not achieve the best overall
balance.

## 11. Overall Best Model

**Selected model: Decision Tree Classifier**

The Decision Tree is selected as the overall best model for this
experiment based primarily on its highest F1-score (0.4303) and highest
MCC (0.3475). These metrics indicate a better balance between precision
and recall and the strongest overall agreement between predicted and
actual classes among the evaluated models.

KNN produced the highest accuracy, but its low recall (0.2316) means it
failed to identify many actual subscribers. Logistic Regression achieved
the highest AUC and recall, but its precision was low. The model choice
therefore depends on the project's objective; for a balanced
classification result, the Decision Tree is the strongest choice based
on these test results.

## 12. Streamlit Application

The Streamlit dashboard provides an interface for evaluating the saved
classification models using an uploaded test CSV file. It is designed to
load saved model pipelines and evaluate the uploaded data **without
retraining**.

The application includes: - CSV test-data upload. - Model selection
through a dropdown. - Evaluation of the available saved models and
comparison of their results. - Display of Accuracy, AUC, Precision,
Recall, F1-score, and MCC. - Confusion matrix and classification
report. - Prediction results for the uploaded records.

The uploaded evaluation CSV must contain all required input features and
the target column `y`. Upload test data only, as required by the
assignment.

**Live Streamlit application:** \[Insert deployed Streamlit Community
Cloud URL\]

## 13. Repository Contents

The repository should contain the application, dependencies,
documentation, test data, notebooks/source code, saved model pipelines,
and required preprocessing artifacts. Example structure:

``` text
bank-marketing-classification/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── bank-full.csv
│   └── cleaned_bank_marketing.csv
│   └── test_data.csv
├── research_notebooks/
│   ├── 01_data_processing.ipynb
│   └── 02_model_building.ipynb
├── models/
│   ├── logistic_regression_tuned_pipeline.joblib
│   ├── decision_tree_tuned_pipeline.joblib
│   ├── knn_tuned_pipeline.joblib
│   ├── gaussian_naive_bayes_tuned_pipeline.joblib
│   └── svm_tuned_pipeline.joblib
└── screenshots/
    └── BITS_Virtual_Lab_Screenshot.png
```

Adjust this example to reflect the files actually included in the GitHub
repository. Do not list files or artifacts that are not present.

## 14. Installation and Execution

1.  Clone the repository:

    ``` bash
    git clone https://github.com/45-Hrishi/bank-marketing-classification.git
    cd bank-marketing-classification
    ```

2.  Install the dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Start the Streamlit application:

    ``` bash
    streamlit run app.py
    ```

4.  Open the local URL shown in the terminal, upload the test CSV, and
    use the dashboard to evaluate the models.

Use the dependency versions recorded and tested in the project
environment in `requirements.txt`, particularly for NumPy, and
scikit-learn compatibility with the saved Joblib artifacts.

## 15. Conclusion

This project implements an end-to-end binary classification workflow
using the UCI Bank Marketing dataset. Five classification algorithms
were trained and tuned, then compared on a held-out test set using
Accuracy, AUC, Precision, Recall, F1-score, and MCC.

Among the evaluated models, the Decision Tree achieved the highest
F1-score and MCC, while KNN achieved the highest accuracy and Logistic
Regression achieved the highest AUC and recall. These differences
demonstrate why multiple evaluation metrics are important when assessing
models on a classification dataset.

The Streamlit application provides an interactive way to upload test
data, evaluate the saved models, and inspect classification results
without retraining.

## 16. References

1.  UCI Machine Learning Repository, [Bank Marketing
    Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).
2.  [Scikit-learn documentation](https://scikit-learn.org/stable/).
3.  [Streamlit documentation](https://docs.streamlit.io/).
4.  [Pandas documentation](https://pandas.pydata.org/docs/).
5.  [NumPy documentation](https://numpy.org/doc/).

------------------------------------------------------------------------
