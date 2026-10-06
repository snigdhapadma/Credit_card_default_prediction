import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Default Prediction")
st.write(
    "Predict whether a credit-card customer is likely to default using "
    "the preprocessing and Random Forest approach from the notebook."
)

DATA_FILE = "Credit Card Defaulter Prediction.csv"
MODEL_FILE = "credit_card_default_model.pkl"


# ---------------------------------------------------------
# Preprocessing - kept consistent with the notebook
# ---------------------------------------------------------
def preprocess_data(df):
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove customer ID
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Convert SEX
    if "SEX" in df.columns:
        if df["SEX"].dtype == "object":
            df["SEX"] = df["SEX"].map({"F": 0, "M": 1})
        else:
            df["SEX"] = df["SEX"].astype(float)

    # Education cleaning
    if "EDUCATION" in df.columns:
        df["EDUCATION"] = df["EDUCATION"].replace({
            0: 4,
            5: 4,
            6: 4
        })

    # Marriage cleaning
    if "MARRIAGE" in df.columns:
        df["MARRIAGE"] = df["MARRIAGE"].replace({
            0: 3
        })

    # Create age group only for consistency, then remove it
    if "AGE" in df.columns:
        bins = [20, 30, 40, 50, 60, 80]
        labels = ["20-29", "30-39", "40-49", "50-59", "60+"]
        df["AGE_GROUP"] = pd.cut(
            df["AGE"],
            bins=bins,
            labels=labels,
            right=False
        )

    # One-hot encode education and marriage
    existing = [c for c in ["EDUCATION", "MARRIAGE"] if c in df.columns]
    if existing:
        df = pd.get_dummies(
            df,
            columns=existing,
            drop_first=True,
            dtype=int
        )

    # Target encoding, if present
    if "default" in df.columns and df["default"].dtype == "object":
        df["default"] = df["default"].map({"N": 0, "Y": 1})

    # Feature engineering from the notebook
    bill_cols = [
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
    ]

    pay_amt_cols = [
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    pay_cols = [
        "PAY_0", "PAY_2", "PAY_3",
        "PAY_4", "PAY_5", "PAY_6"
    ]

    if all(c in df.columns for c in bill_cols):
        df["TOTAL_BILL"] = df[bill_cols].sum(axis=1)
        df["AVG_BILL"] = df[bill_cols].mean(axis=1)
        df["MAX_BILL"] = df[bill_cols].max(axis=1)

    if all(c in df.columns for c in pay_amt_cols):
        df["TOTAL_PAYMENT"] = df[pay_amt_cols].sum(axis=1)
        df["AVG_PAYMENT"] = df[pay_amt_cols].mean(axis=1)
        df["PAYMENT_STD"] = df[pay_amt_cols].std(axis=1)
        df["ZERO_PAYMENT_MONTHS"] = (df[pay_amt_cols] == 0).sum(axis=1)

    if all(c in df.columns for c in pay_cols):
        df["TOTAL_PAY_DELAY"] = df[pay_cols].sum(axis=1)
        df["MAX_PAY_DELAY"] = df[pay_cols].max(axis=1)
        df["NUM_DELAYED_MONTHS"] = (df[pay_cols] > 0).sum(axis=1)
        df["AVG_PAY_DELAY"] = df[pay_cols].mean(axis=1)

    if "BILL_AMT1" in df.columns and "BILL_AMT2" in df.columns:
        df["BILL_CHANGE_1M"] = df["BILL_AMT1"] - df["BILL_AMT2"]

    if (
        "BILL_AMT1" in df.columns
        and "LIMIT_BAL" in df.columns
    ):
        df["CREDIT_UTILIZATION"] = (
            df["BILL_AMT1"] / (df["LIMIT_BAL"] + 1)
        )

    if (
        "AVG_BILL" in df.columns
        and "LIMIT_BAL" in df.columns
    ):
        df["AVG_CREDIT_UTILIZATION"] = (
            df["AVG_BILL"] / (df["LIMIT_BAL"] + 1)
        )

    if (
        "TOTAL_PAYMENT" in df.columns
        and "TOTAL_BILL" in df.columns
    ):
        df["PAYMENT_TO_BILL_RATIO"] = (
            df["TOTAL_PAYMENT"] /
            (df["TOTAL_BILL"].abs() + 1)
        )

    # AGE_GROUP was explicitly removed before model training in the notebook
    if "AGE_GROUP" in df.columns:
        df = df.drop(columns=["AGE_GROUP"])

    return df


# ---------------------------------------------------------
# Load data and train/load model
# ---------------------------------------------------------
@st.cache_data
def load_training_data():
    if not os.path.exists(DATA_FILE):
        return None

    return pd.read_csv(DATA_FILE)


@st.cache_resource
def get_model_and_columns():
    raw = load_training_data()

    if raw is None:
        return None, None

    processed = preprocess_data(raw)

    if "default" not in processed.columns:
        return None, None

    X = processed.drop(columns=["default"])
    y = processed["default"]

    # If the notebook's saved model exists, use it.
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
    else:
        # Exact best Random Forest parameters reported in the notebook.
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X, y)

    return model, X.columns.tolist()


model, feature_columns = get_model_and_columns()


if model is None:
    st.error(
        f"Training data not found. Put '{DATA_FILE}' in the same folder "
        "as this Streamlit file."
    )
    st.stop()





# ---------------------------------------------------------
# Helper for prediction
# ---------------------------------------------------------
def prepare_single_customer(customer):
    raw_customer = pd.DataFrame([customer])
    processed = preprocess_data(raw_customer)

    # Ensure exactly the same columns/order used during training
    processed = processed.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Convert everything to numeric
    processed = processed.apply(pd.to_numeric, errors="coerce").fillna(0)

    return processed


def predict_customer(customer):
    X_input = prepare_single_customer(customer)


    if hasattr(model, "feature_names_in_"):
        X_input = X_input.reindex(
            columns=model.feature_names_in_,
            fill_value=0
        )

    prediction = int(model.predict(X_input)[0])

    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X_input)[0, 1])
    else:
        probability = float(prediction)

    return prediction, probability


st.header("🔍 Customer Default Prediction")
st.caption(
        "Enter the customer's financial and repayment information."
    )

with st.form("prediction_form"):

    st.subheader("Customer Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        limit_bal = st.number_input(
                "Credit Limit (LIMIT_BAL)",
                min_value=1000.0,
                value=50000.0,
                step=1000.0
            )

    with col2:
            sex = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

    with col3:
            age = st.number_input(
                "Age",
                min_value=21,
                max_value=79,
                value=35
            )

    col1, col2 = st.columns(2)

    with col1:
            education = st.selectbox(
                "Education",
                [
                    "Graduate School",
                    "University",
                    "High School",
                    "Other"
                ]
            )

    with col2:
            marriage = st.selectbox(
                "Marriage",
                [
                    "Married",
                    "Single",
                    "Other"
                ]
            )

    st.subheader("Repayment Status")

    pay_cols = [
            "PAY_0", "PAY_2", "PAY_3",
            "PAY_4", "PAY_5", "PAY_6"
        ]

    pay_values = {}

    cols = st.columns(3)

    for i, col in enumerate(pay_cols):
        with cols[i % 3]:
             pay_values[col] = st.number_input(
                    f"{col} - repayment status",
                    min_value=-2,
                    max_value=8,
                    value=0,
                    step=1,
                    help=(
                        "Negative/0 values indicate timely or early payment; "
                        "positive values indicate months of payment delay."
                    )
                )

    st.subheader("Bill Amounts")

    bill_values = {}
    bill_cols = [
            "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
            "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
        ]

    cols = st.columns(3)

    for i, col in enumerate(bill_cols):
            with cols[i % 3]:
                bill_values[col] = st.number_input(
                    col,
                    value=0.0,
                    step=1000.0
                )

    st.subheader("Payment Amounts")

    payment_values = {}
    payment_cols = [
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
            "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
        ]

    cols = st.columns(3)

    for i, col in enumerate(payment_cols):
            with cols[i % 3]:
                payment_values[col] = st.number_input(
                    col,
                    min_value=0.0,
                    value=0.0,
                    step=1000.0
                )

    submitted = st.form_submit_button(
            "Predict Default Risk",
            type="primary"
        )

    if submitted:

        # Education mapping
        education_map = {
            "Graduate School": 1,
            "University": 2,
            "High School": 3,
            "Other": 4
        }

        # Marriage mapping
        marriage_map = {
            "Married": 1,
            "Single": 2,
            "Other": 3
        }

        # Create customer data
        customer = {
            "LIMIT_BAL": limit_bal,
            "SEX": "F" if sex == "Female" else "M",
            "EDUCATION": education_map[education],
            "MARRIAGE": marriage_map[marriage],
            "AGE": age,
            **pay_values,
            **bill_values,
            **payment_values
        }

        # Make prediction
        prediction, probability = predict_customer(customer)

        st.divider()

        # Prediction result
        if prediction == 1:
            st.error("⚠️ Prediction: Customer is likely to DEFAULT")
        else:
            st.success("✅ Prediction: Customer is NOT likely to default")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Default Probability",
                f"{probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Non-Default Probability",
                f"{(1 - probability) * 100:.2f}%"
            )

        with col3:
            risk = (
                "High Risk"
                if probability >= 0.50
                else "Lower Risk"
            )

            st.metric("Risk Level", risk)

        st.progress(probability)

        if probability >= 0.70:
            st.warning(
                "High predicted default probability. "
                "This customer may require additional credit-risk review."
            )

        elif probability >= 0.50:
            st.warning(
                "Moderate-to-high predicted default probability."
            )

        else:
            st.info(
                "The model predicts a lower probability of default."
            )