# 🛡️ FraudShield AI

## AI-Powered Financial Transaction Fraud Detection using XGBoost & Explainable AI

**FraudShield AI** is a machine-learning project designed to identify potentially fraudulent financial transactions using **XGBoost**. It combines transaction feature engineering, class-imbalance handling, threshold optimization, explainable AI, risk scoring, and a **Streamlit web application**.

> **Project type:** Educational / portfolio project  
> **Primary model:** XGBoost Classifier  
> **Target:** `isFraud`

---

## 🎯 Project Objective

Fraudulent transactions generally form a small minority of financial transactions, making fraud detection an imbalanced classification problem.

FraudShield AI aims to:

- Detect potentially fraudulent transactions.
- Handle class imbalance during model training.
- Generate fraud probabilities in addition to binary predictions.
- Optimize the classification threshold using validation data.
- Evaluate performance with metrics suitable for imbalanced classification.
- Explain model behavior using SHAP.
- Present predictions through understandable risk levels.
- Provide an interactive Streamlit interface.

---

## 🔄 Machine Learning Workflow

```text
Transaction Dataset
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
Fraud Probability & Risk Scoring
        ↓
Streamlit Application
```

---

## 📊 Dataset

The project works with financial transaction fields including:

| Feature | Description |
|---|---|
| `step` | Transaction time step |
| `type` | Type of transaction |
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

Because the target is imbalanced, **accuracy is not used as the only performance measure**.

---

## 🔍 Exploratory Data Analysis

EDA examines:

- Fraud vs. legitimate transaction distribution
- Fraud patterns across transaction types
- Transaction amount characteristics
- Class imbalance
- Transaction behavior relevant to feature engineering

---

## 🧠 Feature Engineering

FraudShield AI creates behavioral features from the original transaction information.

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

Transaction type is one-hot encoded before modeling.

---

## ⚙️ Data Preparation

The pipeline includes:

- Handling missing target labels
- Removing high-cardinality sender and receiver identifiers from the baseline model
- Excluding the existing `isFlaggedFraud` rule-based indicator
- One-hot encoding transaction type
- Separating features and target
- Stratified train/validation/test splitting

---

## ⚖️ Handling Class Imbalance

XGBoost uses `scale_pos_weight`, calculated from the training data:

```python
scale_pos_weight = negative_samples / positive_samples
```

This gives additional importance to learning patterns associated with the minority fraud class.

---

## 🤖 XGBoost Model

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

XGBoost is the primary model used to learn nonlinear patterns and interactions in the structured transaction data.

---

## 🎯 Threshold Optimization

Instead of automatically using a `0.50` classification threshold, FraudShield AI evaluates the **precision-recall trade-off on validation data**.

The optimized threshold is then applied to the untouched test set for final evaluation.

---

## 📈 Model Evaluation

The project evaluates the model using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC
- PR-AUC

For an imbalanced fraud-detection problem, **Recall, F1-Score and PR-AUC** are especially useful alongside accuracy.

> Final metric values should be taken directly from the executed final notebook rather than manually estimated.

---

## 🔎 Explainable AI with SHAP

FraudShield AI uses **SHAP (SHapley Additive exPlanations)** for model interpretation.

**Global explainability** helps identify which features contribute most strongly to predictions across the dataset.

**Local explainability** helps explain why an individual transaction receives a particular fraud prediction.

---

## 🚨 Fraud Risk Assessment

The trained model generates a fraud probability. For demonstration, the Streamlit interface also maps probabilities into:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

These are project-defined presentation bands, not universal financial-industry thresholds.

---

## 💻 Streamlit Application

The interactive interface accepts:

- Transaction type
- Transaction amount
- Sender balance before and after the transaction
- Receiver balance before and after the transaction
- Transaction step

It recreates the required engineered features and displays the **fraud probability, risk level, model classification, transaction summary, and engineered feature information**.

---

## 💾 Saved Model Artifacts

The XGBoost model is saved in its native format:

```python
model.save_model("fraud_xgboost.json")
```

Supporting artifacts:

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

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainable AI:** SHAP
- **Visualization:** Matplotlib
- **Web Application:** Streamlit
- **Model Persistence:** XGBoost native model format, Joblib

---

## 🚀 Run FraudShield AI Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd FraudShield-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate it on Windows

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Start Streamlit

```bash
python -m streamlit run app.py
```

---

## 📦 Requirements

```text
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
shap
matplotlib
```

---

## ✨ Key Highlights

- End-to-end financial fraud detection workflow
- XGBoost classification
- Class-imbalance handling
- Transaction-based feature engineering
- Validation-based threshold optimization
- Fraud probability estimation
- Multiple evaluation metrics
- Global and local SHAP explainability
- Interactive Streamlit interface
- Reusable model artifacts

---

## 🔮 Future Improvements

- Real-time transaction scoring
- Cost-sensitive threshold optimization
- Additional behavioral features
- Hyperparameter optimization
- Model monitoring and drift detection
- SHAP explanations inside Streamlit
- Batch CSV transaction analysis
- Fraud investigation dashboard
- Cloud deployment

---

## ⚠️ Disclaimer

**FraudShield AI is an educational portfolio project.**

Predictions and risk levels should not be used for real-world financial, banking, customer-account, or fraud-investigation decisions without appropriate validation, governance, security controls, and human review.

---

## 👩‍💻 Author

**Radhika Sihra**

B.Tech — Electronics & Telecommunication Engineering  
Aspiring Data Scientist | Machine Learning | Deep Learning | Generative AI

---

## 🛡️ FraudShield AI

**Detect suspicious transactions. Explain model decisions. Support smarter fraud analysis.**
