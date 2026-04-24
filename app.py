import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Loan Approval System", layout="centered")

# =========================================================
# SYNTHETIC DATA (REALISTIC SIMULATION)
# =========================================================
@st.cache_data
def create_data():
    np.random.seed(42)
    n = 2000

    df = pd.DataFrame({
        "Income": np.random.randint(20000, 150000, n),
        "Age": np.random.randint(21, 60, n),
        "LoanAmount": np.random.randint(5000, 50000, n),
        "CreditScore": np.random.randint(300, 850, n),
    })

    # Realistic approval logic
    df["Approved"] = (
        (df["Income"] > 50000) &
        (df["CreditScore"] > 600) &
        (df["LoanAmount"] < df["Income"] * 0.5)
    ).astype(int)

    return df

df = create_data()

# =========================================================
# TRAIN MODEL
# =========================================================
X = df.drop("Approved", axis=1)
y = df["Approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# =========================================================
# EVALUATION
# =========================================================
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def calculate_risk(prob):
    return round((1 - prob) * 100, 2)

def calculate_emi(loan, rate=0.1, years=2):
    months = years * 12
    emi = (loan * rate/12 * (1 + rate/12)**months) / ((1 + rate/12)**months - 1)
    return emi

# =========================================================
# UI HEADER
# =========================================================
st.title("🏦 Smart Loan Approval System")
st.write(f"Model Accuracy: {round(acc,2)}")

st.markdown("---")

# =========================================================
# USER INPUT
# =========================================================
st.subheader("📋 Enter Applicant Details")

income = st.number_input("💰 Income (₹)", value=60000)
age = st.slider("🎂 Age", 18, 65, 30)
loan = st.number_input("🏦 Loan Amount (₹)", value=10000)
credit = st.slider("📊 Credit Score", 300, 850, 650)

# =========================================================
# PREDICTION
# =========================================================
input_df = pd.DataFrame([{
    "Income": income,
    
    "Age": age,
    "LoanAmount": loan,
    "CreditScore": credit
}])

if st.button("🔍 Evaluate Loan"):

    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    risk = calculate_risk(prob)
    emi = calculate_emi(loan)

    st.markdown("---")
    st.subheader("📊 Analysis")

    col1, col2 = st.columns(2)

    col1.metric("Approval Probability", f"{round(prob*100,1)}%")
    col2.metric("Risk Score", f"{risk}/100")

    st.metric("Estimated EMI", f"₹ {round(emi,2)}")

    st.progress(int(prob * 100))

    # =====================================================
    # BUSINESS LOGIC (REAL DECISION ENGINE)
    # =====================================================
    reasons = []

    if income < 40000:
        reasons.append("Low income")

    if credit < 600:
        reasons.append("Low credit score")

    if loan > income * 0.6:
        reasons.append("Loan too high compared to income")

    if emi > income * 0.3:
        reasons.append("EMI exceeds safe limit (30% income)")

    st.markdown("---")

    # =====================================================
    # FINAL DECISION
    # =====================================================
    if risk < 40 and len(reasons) == 0:
        st.success("✅ Loan Approved (Low Risk)")

    elif risk < 70:
        st.warning("⚠️ Conditional Approval")

    else:
        st.error("❌ Loan Rejected (High Risk)")

    # =====================================================
    # EXPLANATION
    # =====================================================
    if reasons:
        st.subheader("⚠️ Reasons for Decision")
        for r in reasons:
            st.write(f"- {r}")

    else:
        st.subheader("✅ Strong Financial Profile")
        st.write("No major risk factors detected.")
