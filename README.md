# 🛡️ FraudShield AI

## AI-Powered Financial Transaction Fraud Detection using XGBoost & Explainable AI

**FraudShield AI** is an end-to-end machine learning project designed to identify potentially fraudulent financial transactions using **XGBoost**.

The system combines **transaction feature engineering, class-imbalance handling, threshold optimization, explainable AI with SHAP, fraud probability estimation, risk scoring, and an interactive Streamlit application**.

> **Primary Model:** XGBoost Classifier  
> **Target Variable:** `isFraud`  
> **Domain:** Financial Fraud Detection  
> **Application:** Machine Learning + Explainable AI

---

## 🎯 Project Objective

Financial fraud detection is a challenging machine learning problem because fraudulent transactions generally represent only a small percentage of total transactions.

FraudShield AI was developed to:

- Detect potentially fraudulent financial transactions
- Handle highly imbalanced transaction data
- Engineer transaction-behavior features
- Generate fraud probabilities in addition to binary predictions
- Optimize the classification threshold using validation data
- Evaluate the model using imbalance-aware metrics
- Explain model predictions using **SHAP**
- Convert model probabilities into understandable risk levels
- Provide predictions through an interactive **Streamlit application**

---

## 🔄 Machine Learning Workflow

```text
Financial Transaction Dataset
            ↓
Data Understanding & Cleaning
            ↓
Exploratory Data Analysis
            ↓
Feature Engineering
            ↓
Feature Preparation & Encoding
            ↓
Train / Validation / Test Split
            ↓
Class Imbalance Handling
            ↓
XGBoost Model Training
            ↓
Threshold Optimization
            ↓
Final Model Evaluation
            ↓
Feature Importance + SHAP
            ↓
Fraud Probability Estimation
            ↓
Risk Classification
            ↓
Streamlit Application
```

---

## 📊 Dataset

The project works with financial transaction information including:

| Feature | Description |
|---|---|
| `step` | Transaction time step |
| `type` | Transaction type |
| `amount` | Transaction amount |
| `nameOrig` | Sender identifier |
| `oldbalanceOrg` | Sender balance before transaction |
| `newbalanceOrig` | Sender balance after transaction |
| `nameDest` | Receiver identifier |
| `oldbalanceDest` | Receiver balance before transaction |
| `newbalanceDest` | Receiver balance after transaction |
| `isFraud` | Fraud target |
| `isFlaggedFraud` | Existing rule-based fraud flag |

### Target Variable

```text
0 → Legitimate Transaction
1 → Fraudulent Transaction
```

Because fraudulent transactions are the minority class, **accuracy alone is not sufficient for evaluating the model**.

---

## 🔍 Exploratory Data Analysis

The analysis focuses on:

- Fraud vs. legitimate transaction distribution
- Class imbalance
- Fraud patterns across transaction types
- Transaction amount characteristics
- Transaction behavior relevant to feature engineering

---

## 🧠 Feature Engineering

FraudShield AI creates additional behavioral features from the original transaction information.

### Sender Balance Error

```python
orig_balance_error = oldbalanceOrg - amount - newbalanceOrig
```

### Receiver Balance Error

```python
dest_balance_error = oldbalanceDest + amount - newbalanceDest
```

### Additional Engineered Features

```text
orig_zero_balance
dest_zero_balance
amount_to_orig_balance
hour
```

Transaction `type` is one-hot encoded before model training.

---

## ⚙️ Data Preparation

The machine learning pipeline includes:

- Handling missing target labels
- Removing high-cardinality sender and receiver identifiers from the baseline model
- Excluding the existing `isFlaggedFraud` rule-based indicator
- One-hot encoding transaction types
- Separating features and target
- Stratified train, validation, and test splitting

---

## ⚖️ Class Imbalance Handling

FraudShield AI handles class imbalance using XGBoost's `scale_pos_weight`.

```python
scale_pos_weight = negative_samples / positive_samples
```

This gives greater importance to the minority fraud class during model training.

---

## 🤖 XGBoost Fraud Detection Model

The primary machine learning algorithm used in FraudShield AI is **XGBoost**.

```python
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
```

XGBoost learns nonlinear patterns and interactions within structured financial transaction data.

---

## 🎯 Threshold Optimization

Instead of automatically using the standard `0.50` classification threshold, FraudShield AI evaluates the **precision-recall trade-off on validation data**.

The selected threshold is then applied to the untouched test dataset for final evaluation.

---

## 📈 Model Evaluation

The model is evaluated using:

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall percentage of correct predictions |
| **Precision** | Reliability of fraud predictions |
| **Recall** | Ability to identify fraudulent transactions |
| **F1-Score** | Balance between precision and recall |
| **Confusion Matrix** | Breakdown of prediction outcomes |
| **ROC-AUC** | Ranking performance across thresholds |
| **PR-AUC** | Precision-recall performance |

For an imbalanced fraud-detection problem, **Recall, F1-Score, and PR-AUC** are particularly useful alongside accuracy.

> Final metric values should be taken directly from the executed final notebook rather than manually estimated.

---

## 🔎 Explainable AI with SHAP

FraudShield AI incorporates **SHAP (SHapley Additive exPlanations)** to improve model interpretability.

### 🌍 Global Explainability

Global SHAP analysis helps identify which features contribute most strongly to model predictions across the dataset.

### 🔍 Local Explainability

Local SHAP explanations help understand why the model assigns a particular fraud prediction to an individual transaction.

---

## 🚨 Fraud Risk Assessment

The trained model generates a **fraud probability** for each transaction.

For presentation in the Streamlit application, probabilities are mapped into four project-defined risk levels:

```text
🟢 LOW
🟡 MEDIUM
🟠 HIGH
🔴 CRITICAL
```

> These risk bands are project-defined presentation categories and are not universal financial-industry thresholds.

---

## 💻 Interactive Streamlit Application

FraudShield AI includes an interactive **Streamlit web interface**.

### Application Inputs

- Transaction type
- Transaction amount
- Sender balance before transaction
- Sender balance after transaction
- Receiver balance before transaction
- Receiver balance after transaction
- Transaction step

### Application Output

- Fraud probability
- Fraud risk level
- Model classification
- Transaction summary
- Engineered feature information

---

## 💾 Model Artifacts

The trained XGBoost model is stored using XGBoost's native model format:

```python
model.save_model("fraud_xgboost.json")
```

Supporting model artifacts include:

```text
fraud_xgboost.json
model_features.pkl
threshold.pkl
```

---

## 📁 Project Structure

```text
FraudShield-AI/
│
├── app.py
├── fraud_detection.ipynb
├── fraud.csv
├── fraud_xgboost.json
├── model_features.pkl
├── threshold.pkl
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| **Programming** | Python |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Explainable AI** | SHAP |
| **Visualization** | Matplotlib |
| **Web Application** | Streamlit |
| **Model Persistence** | XGBoost Native Format, Joblib |

---

## ✨ Key Project Highlights

- End-to-end financial transaction fraud-detection pipeline
- XGBoost-based binary classification
- Class-imbalance handling using `scale_pos_weight`
- Transaction behavior-based feature engineering
- Stratified train/validation/test splitting
- Validation-based classification threshold optimization
- Fraud probability estimation
- Risk-level classification
- Multiple imbalance-aware evaluation metrics
- Global and local explainability using SHAP
- Interactive Streamlit web application
- Reusable trained model artifacts

---

## 🔮 Future Improvements

- Real-time transaction scoring
- Cost-sensitive threshold optimization
- Additional behavioral fraud features
- Automated hyperparameter optimization
- Model monitoring and data-drift detection
- SHAP explanations directly inside Streamlit
- Batch CSV transaction analysis
- Fraud investigation dashboard
- Cloud deployment

---

## ⚠️ Disclaimer

**FraudShield AI is an educational portfolio project.**

Predictions and risk levels generated by this project should not be used for real-world banking, financial, customer-account, or fraud-investigation decisions without appropriate validation, governance, security controls, and human review.

---

## 👩‍💻 Author

**Radhika Sihra**

B.Tech — Electronics & Telecommunication Engineering

**Aspiring Data Scientist | Machine Learning | Deep Learning | Generative AI**

---

# 🛡️ FraudShield AI

### Detect suspicious transactions. Explain model decisions. Support smarter fraud analysis.
