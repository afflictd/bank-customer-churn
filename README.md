# Bank Customer Churn Classification

A machine learning project focused on predicting bank customer churn using classical classification algorithms.

The main goal of this project was not only to achieve good predictive performance, but also to compare different classification approaches and practice proper model evaluation, cross-validation, preprocessing pipelines, hyperparameter tuning, and data leakage prevention.

---

## Project Goal

Customer churn is an important business problem for banks because retaining an existing customer is often more cost-effective than acquiring a new one.

The objective of this project is to predict whether a customer is likely to leave the bank based on demographic and account-related information.

This is a binary classification problem:

- `0` — customer stays
- `1` — customer leaves

The target variable is imbalanced, with approximately 20% of customers belonging to the churn class.

---

## Models

The following classification algorithms were explored and compared:

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest

These models were selected because they represent different approaches to classification:

- Logistic Regression provides a simple and interpretable linear baseline.
- KNN is based on distances between observations.
- SVM allows both linear and nonlinear decision boundaries.
- Decision Tree provides interpretable nonlinear rules.
- Random Forest reduces the variance of individual trees using an ensemble.

---

## Workflow

The project follows the following workflow:

1. Exploratory Data Analysis
2. Data Cleaning
3. Feature Selection
4. Train/Test Split with Stratification
5. Feature Preprocessing
6. Model Training
7. Cross-Validation
8. Hyperparameter Experiments
9. Hyperparameter Tuning
10. Model Comparison
11. Final Evaluation on a Held-Out Test Set

---

## Data Preprocessing

The preprocessing pipeline includes:

- Standardization of numerical features using `StandardScaler`
- One-hot encoding of categorical features using `OneHotEncoder`
- Stratified train/test split to preserve the target class distribution
- `Pipeline` and `ColumnTransformer` to prevent data leakage during cross-validation

Preprocessing is fitted only on training folds during cross-validation.

---

## Data Leakage

One of the features, `Complain`, had an extremely strong relationship with the target variable.

Such a relationship may indicate target leakage if information about the complaint becomes available only during or after the customer churn process.

For this reason, the feature was excluded from model training.

Other non-informative identifier features such as:

- `RowNumber`
- `CustomerId`
- `Surname`

were also removed.

---

## Evaluation Metric

The primary evaluation metric is **F1-score**.

Because the dataset is imbalanced, accuracy alone may be misleading. For example, a classifier predicting the majority class for every observation could still achieve approximately 80% accuracy.

F1-score was selected because both precision and recall are important in the churn prediction task.

- Low recall means the bank fails to identify many customers who are actually going to leave.
- Low precision means the bank spends resources on customers who were not actually going to leave.

---

## Cross-Validation Results

Baseline models were compared using stratified cross-validation.

| Model | Mean CV F1 |
|---|---:|
| Logistic Regression | 0.488 |
| K-Nearest Neighbors | 0.438 |
| Support Vector Machine | 0.589 |
| Decision Tree | 0.477 |
| Random Forest | 0.522 |

SVM demonstrated the strongest baseline performance.

After hyperparameter tuning, Random Forest achieved the best cross-validation performance.

---

## Hyperparameter Experiments

### K-Nearest Neighbors

Different values of `k` were tested using cross-validation.

Small values of `k` produced high training performance but were more prone to overfitting.

For example:

| k | Train F1 | CV F1 |
|---:|---:|---:|
| 1 | 1.000 | 0.433 |
| 3 | 0.695 | 0.443 |
| 50 | 0.320 | 0.266 |

Feature scaling also had a significant effect on KNN performance.

Without scaling, features with large numerical ranges dominate distance calculations.

---

### Decision Tree

Different tree depths were tested to observe the bias-variance trade-off.

Deep trees achieved almost perfect training performance while validation performance decreased.

For example:

| max_depth | Train F1 | CV F1 |
|---:|---:|---:|
| 5 | 0.546 | 0.521 |
| 20 | 0.988 | 0.485 |
| None | 1.000 | 0.486 |

This demonstrates strong overfitting in unrestricted decision trees.

---

### Support Vector Machine

The influence of `C` and `gamma` was investigated.

For large values of `C`, training performance increased significantly while cross-validation performance decreased.

For example:

| C | Train F1 | CV F1 |
|---:|---:|---:|
| 10 | 0.770 | 0.548 |
| 100 | 0.963 | 0.506 |

Very large `gamma` values also caused strong overfitting.

For example:

```text
gamma = 1
Train F1 = 1.000
CV F1 ≈ 0.039
```

Feature scaling was especially important for SVM.

---

### Random Forest

Random Forest demonstrated more stable validation performance than a single Decision Tree.

Increasing the number of trees improved performance initially, but the improvement eventually reached a plateau.

For example:

| n_estimators | CV F1 |
|---:|---:|
| 50 | 0.544 |
| 100 | 0.557 |
| 300 | 0.566 |
| 500 | 0.566 |
| 1000 | 0.567 |

This shows that adding more trees eventually provides very limited improvement while increasing computational cost.

---

## Hyperparameter Tuning

Hyperparameters were selected using `GridSearchCV` with preprocessing included inside the pipeline.

The best Random Forest configuration achieved approximately:

```text
CV F1 ≈ 0.617
```

The selected parameters included:

```text
max_depth = 10
min_samples_split = 10
n_estimators = 100
```

The final model was selected using cross-validation before evaluating it on the test set.

---

## Final Model

Random Forest was selected as the final model based on cross-validation performance.

The held-out test set was used only after model selection and hyperparameter tuning were completed.

### Test Results

| Metric | Score |
|---|---:|
| Accuracy | 0.835 |
| Precision | 0.581 |
| Recall | 0.676 |
| F1-score | 0.625 |

The model identifies approximately 68% of customers who actually leave the bank while maintaining a precision of approximately 58%.

---

## Feature Importance

Random Forest feature importance was used to investigate which variables contributed most to model decisions.

Feature importance should not be interpreted as evidence of causality. A feature may be strongly associated with churn without directly causing it.

---

## Decision Boundaries

To better understand the behavior of different classification algorithms, decision boundaries were visualized using two numerical features.

This experiment demonstrates how different models construct classification regions:

- Logistic Regression produces a linear boundary.
- KNN creates a local and potentially irregular boundary.
- RBF SVM can create nonlinear smooth boundaries.
- Decision Trees create axis-aligned regions.
- Random Forest combines multiple tree-based boundaries.

---

## Key Learnings

This project was used to practice and understand:

- Binary classification
- Class imbalance
- Precision, Recall and F1-score
- Stratified train/test splitting
- Cross-validation
- Data leakage
- Scikit-learn pipelines
- Feature scaling
- One-hot encoding
- Bias-variance trade-off
- Model overfitting and underfitting
- Hyperparameter tuning
- Decision boundaries
- Feature importance
- Model comparison

---

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

---

## Repository Structure

```text
bank-customer-churn/
│
├── images/
│
├── notebooks/
│   └── bank_customer_churn.ipynb
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/afflictd/bank-customer-churn.git
cd bank-customer-churn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
notebooks/bank_customer_churn.ipynb
```

and run all cells.

---

## Author

GitHub: [afflictd](https://github.com/afflictd)