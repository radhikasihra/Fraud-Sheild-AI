from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
st.set_page_config(
    page_title="FRAUDSHEILD AI",
    page_icon="🔍",
    layout="wide"
)
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_xgboost.json"
FEATURES_PATH = BASE_DIR / "model_features.pkl"
THRESHOLD_PATH = BASE_DIR / "threshold.pkl"
@st.cache_resource
def load_model():

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    features = joblib.load(FEATURES_PATH)
    threshold = joblib.load(THRESHOLD_PATH)

    return model, features, threshold


try:
    model, model_features, threshold = load_model()

except Exception as e:

    st.error(
        "Unable to load the trained model. "
        "Make sure fraud_xgboost.json, model_features.pkl "
        "and threshold.pkl are in the same folder as app.py."
    )

    st.exception(e)
    st.stop()

st.title("🔍 FRAUD SHEILD AI")

st.markdown(
    """
    This application uses an **XGBoost Machine Learning model**
    to analyze financial transactions and estimate their
    probability of being fraudulent.

    Enter the transaction information below and select
    **Analyze Transaction** to generate an AI-based fraud
    risk assessment.
    """
)

st.divider()


# ============================================================
# 5. TRANSACTION INFORMATION
# ============================================================

st.subheader("💳 Transaction Information")

col1, col2 = st.columns(2)

with col1:

    transaction_type = st.selectbox(
        "Transaction Type",
        [
            "CASH_IN",
            "CASH_OUT",
            "DEBIT",
            "PAYMENT",
            "TRANSFER"
        ]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    step = st.number_input(
        "Transaction Step",
        min_value=1,
        value=1,
        step=1,
        help="Time step associated with the transaction."
    )


with col2:

    oldbalanceOrg = st.number_input(
        "Sender Balance Before Transaction",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    newbalanceOrig = st.number_input(
        "Sender Balance After Transaction",
        min_value=0.0,
        value=4000.0,
        step=100.0
    )


# ============================================================
# 6. RECEIVER INFORMATION
# ============================================================

st.subheader("🏦 Receiver Information")

col3, col4 = st.columns(2)

with col3:

    oldbalanceDest = st.number_input(
        "Receiver Balance Before Transaction",
        min_value=0.0,
        value=0.0,
        step=100.0
    )


with col4:

    newbalanceDest = st.number_input(
        "Receiver Balance After Transaction",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )


st.divider()


# ============================================================
# 7. FEATURE ENGINEERING
# ============================================================

def prepare_transaction():

    # Original transaction variables
    data = {
        "step": step,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
    }

    input_df = pd.DataFrame([data])

    # --------------------------------------------------------
    # Feature 1: Sender balance inconsistency
    # --------------------------------------------------------

    input_df["orig_balance_error"] = (
        input_df["oldbalanceOrg"]
        - input_df["amount"]
        - input_df["newbalanceOrig"]
    )

    # --------------------------------------------------------
    # Feature 2: Receiver balance inconsistency
    # --------------------------------------------------------

    input_df["dest_balance_error"] = (
        input_df["oldbalanceDest"]
        + input_df["amount"]
        - input_df["newbalanceDest"]
    )

    # --------------------------------------------------------
    # Feature 3: Sender zero-balance indicator
    # --------------------------------------------------------

    input_df["orig_zero_balance"] = (
        input_df["oldbalanceOrg"] == 0
    ).astype(int)

    # --------------------------------------------------------
    # Feature 4: Receiver zero-balance indicator
    # --------------------------------------------------------

    input_df["dest_zero_balance"] = (
        input_df["oldbalanceDest"] == 0
    ).astype(int)

    # --------------------------------------------------------
    # Feature 5: Amount relative to sender balance
    # --------------------------------------------------------

    input_df["amount_to_orig_balance"] = (
        input_df["amount"]
        / (input_df["oldbalanceOrg"] + 1)
    )

    # --------------------------------------------------------
    # Feature 6: Transaction hour
    # Same calculation used during model training
    # --------------------------------------------------------

    input_df["hour"] = input_df["step"] % 24

    # --------------------------------------------------------
    # One-hot encode transaction type
    # --------------------------------------------------------

    transaction_types = [
        "CASH_IN",
        "CASH_OUT",
        "DEBIT",
        "PAYMENT",
        "TRANSFER"
    ]

    for transaction in transaction_types:

        input_df[f"type_{transaction}"] = int(
            transaction_type == transaction
        )

    # --------------------------------------------------------
    # Match exact training feature order
    # --------------------------------------------------------

    input_df = input_df.reindex(
        columns=model_features,
        fill_value=0
    )

    return input_df


# ============================================================
# 8. FRAUD RISK LEVEL
# ============================================================

def get_risk_level(probability):

    if probability >= 0.80:
        return "CRITICAL"

    elif probability >= 0.60:
        return "HIGH"

    elif probability >= 0.30:
        return "MEDIUM"

    else:
        return "LOW"


# ============================================================
# 9. MODEL PREDICTION
# ============================================================

if st.button(
    "🔍 Analyze Transaction",
    type="primary",
    use_container_width=True
):

    input_df = prepare_transaction()

    # Predict fraud probability
    fraud_probability = model.predict_proba(
        input_df
    )[0][1]

    # Apply optimized threshold from model training
    fraud_prediction = int(
        fraud_probability >= threshold
    )

    # Convert probability to readable risk category
    risk_level = get_risk_level(
        fraud_probability
    )

    st.subheader("📊 Fraud Risk Assessment")

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

    with metric2:

        st.metric(
            "Risk Level",
            risk_level
        )

    with metric3:

        prediction_text = (
            "Fraud"
            if fraud_prediction == 1
            else "Legitimate"
        )

        st.metric(
            "Model Classification",
            prediction_text
        )


    # ========================================================
    # 10. FINAL DECISION
    # ========================================================

    if fraud_prediction == 1:

        st.error(
            "⚠️ Potential Fraud Detected — "
            "This transaction has been flagged "
            "for further investigation."
        )

    else:

        st.success(
            "✅ Transaction Not Flagged — "
            "The model classified this transaction "
            "as legitimate."
        )


    # ========================================================
    # 11. PROBABILITY VISUALIZATION
    # ========================================================

    st.markdown("#### Fraud Probability")

    st.progress(
        int(fraud_probability * 100)
    )

    st.write(
        f"**{fraud_probability * 100:.2f}% probability of fraud**"
    )


    # ========================================================
    # 12. TRANSACTION SUMMARY
    # ========================================================

    with st.expander("📋 View Transaction Summary"):

        summary = pd.DataFrame(
            {
                "Transaction Detail": [
                    "Transaction Type",
                    "Amount",
                    "Sender Balance Before",
                    "Sender Balance After",
                    "Receiver Balance Before",
                    "Receiver Balance After",
                    "Transaction Step"
                ],

                "Value": [
                    transaction_type,
                    amount,
                    oldbalanceOrg,
                    newbalanceOrig,
                    oldbalanceDest,
                    newbalanceDest,
                    step
                ]
            }
        )

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # 13. ENGINEERED FEATURES
    # ========================================================

    with st.expander(
        "🧠 View AI Engineered Features"
    ):

        orig_balance_error = (
            oldbalanceOrg
            - amount
            - newbalanceOrig
        )

        dest_balance_error = (
            oldbalanceDest
            + amount
            - newbalanceDest
        )

        amount_balance_ratio = (
            amount / (oldbalanceOrg + 1)
        )

        engineered_features = pd.DataFrame(
            {
                "Feature": [
                    "Sender Balance Error",
                    "Receiver Balance Error",
                    "Amount / Sender Balance",
                    "Transaction Hour"
                ],

                "Value": [
                    orig_balance_error,
                    dest_balance_error,
                    amount_balance_ratio,
                    step % 24
                ]
            }
        )

        st.dataframe(
            engineered_features,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 14. MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("🤖 About the AI Model"):

    st.markdown(
        """
        **Model:** XGBoost Classifier

        **Purpose:** Financial transaction fraud detection

        **Class Imbalance Handling:** `scale_pos_weight`

        **Decision Strategy:** Optimized classification threshold

        **Model Evaluation:** Accuracy, Precision, Recall,
        F1-Score, ROC-AUC and PR-AUC

        **Explainability:** SHAP analysis is used in the
        model-development notebook to understand important
        fraud indicators.
        """
    )


# ============================================================
# 15. DISCLAIMER
# ============================================================

st.caption(
    "Portfolio project for educational and demonstration purposes. "
    "Predictions should not be used as real financial decisions."
)