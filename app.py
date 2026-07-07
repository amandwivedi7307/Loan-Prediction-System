import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Loan Prediction System",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("loan_model.pkl")

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main{
    background:#F4F6F9;
}

.title{
    font-size:45px;
    color:#0E76A8;
    font-weight:bold;
    text-align:center;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:10px;
    font-size:20px;
    background:#0E76A8;
    color:white;
}

.stButton>button:hover{
    background:#085A80;
    color:white;
}

.css-1d391kg{
    background:#ffffff;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.image(
    "https://img.icons8.com/color/96/bank-building.png",
    width=80
)

st.sidebar.title("Loan Prediction")

st.sidebar.write("---")

st.sidebar.markdown("## 👨‍💻 Developer")

st.sidebar.write("**Aman Dwivedi**")

st.sidebar.write("B.Tech AIML")

st.sidebar.write("---")

st.sidebar.markdown(
"[🐙 GitHub](https://github.com/amandwivedi7307)"
)

st.sidebar.markdown(
"[💼 LinkedIn](https://linkedin.com)"
)

st.sidebar.write("---")

st.sidebar.success("Machine Learning Project")

# -----------------------------
# Title
# -----------------------------

st.markdown(
'<p class="title">🏦 Loan Prediction Dashboard</p>',
unsafe_allow_html=True
)

st.markdown(
'<p class="subtitle">Predict whether a customer is eligible for a loan using Machine Learning.</p>',
unsafe_allow_html=True
)

st.write("")

# -----------------------------
# Input Form
# -----------------------------

left, right = st.columns(2)

with left:

    gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    married = st.selectbox(
        "Married",
        ["Yes","No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["0","1","2","3+"]
    )

    education = st.selectbox(
        "Education",
        ["Graduate","Not Graduate"]
    )

    self_emp = st.selectbox(
        "Self Employed",
        ["Yes","No"]
    )

    property_area = st.selectbox(
        "Property Area",
        ["Urban","Semiurban","Rural"]
    )

with right:

    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0,
        value=5000
    )

    coapplicant_income = st.number_input(
        "Co-Applicant Income",
        min_value=0,
        value=0
    )

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1,
        value=120
    )

    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=12,
        value=360
    )

    credit_history = st.selectbox(
        "Credit History",
        [1.0,0.0]
    )

st.write("")

predict = st.button("🔍 Predict Loan Status")

# -----------------------------
# Prediction
# -----------------------------
if predict:

    # -----------------------------
    # Prepare Input
    # -----------------------------

    total_income = applicant_income + coapplicant_income
    loan_log = np.log(loan_amount)

    input_df = pd.DataFrame({
        "Gender": [gender],
        "Married": [married],
        "Dependents": [dependents],
        "Education": [education],
        "Self_Employed": [self_emp],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area],
        "Total_Income": [total_income],
        "LoanAmount_log": [loan_log]
    })

    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df).max() * 100

    st.write("")

    # -----------------------------
    # Result
    # -----------------------------

    if prediction == "Y":

        st.balloons()

        st.success("🎉 Congratulations!")

        st.markdown("## ✅ Loan Approved")

    else:

        st.error("❌ Loan Rejected")

        st.warning("The application has a lower probability of approval.")

    # -----------------------------
    # Metrics
    # -----------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Model Accuracy",
            "78.38%"
        )

    with col2:

        st.metric(
            "Approval Probability",
            f"{probability:.2f}%"
        )

    # -----------------------------
    # Charts
    # -----------------------------

    st.write("---")

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Applicant Income")

        fig, ax = plt.subplots(figsize=(5,3))

        ax.bar(
            ["Applicant", "Co-Applicant"],
            [applicant_income, coapplicant_income]
        )

        ax.set_ylabel("Income")

        st.pyplot(fig)

    with c2:

        st.subheader("Loan Amount")

        fig2, ax2 = plt.subplots(figsize=(5,3))

        ax2.bar(
            ["Loan"],
            [loan_amount]
        )

        ax2.set_ylabel("Amount")

        st.pyplot(fig2)

    # -----------------------------
    # Credit History
    # -----------------------------

    st.write("---")

    st.subheader("Credit History")

    if credit_history == 1.0:

        st.success("✅ Good Credit History")

    else:

        st.error("❌ Poor Credit History")

    # -----------------------------
    # Input Summary
    # -----------------------------

    st.write("---")

    st.subheader("Application Summary")

    st.dataframe(input_df, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------

st.write("---")

st.markdown(
"""
<div style="text-align:center">

<h4>🏦 Loan Prediction System</h4>

Developed by <b>Aman Dwivedi</b>

<a href="https://github.com/amandwivedi7307" target="_blank">
GitHub
</a>

</div>
""",
unsafe_allow_html=True
)