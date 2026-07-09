import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from assets import income_chart, loan_chart
from report import create_pdf
import seaborn as sns
import plotly.graph_objects as go


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("loan_model.pkl")
performance = joblib.load("performance.pkl")




# -----------------------------
# Custom CSS
# -----------------------------

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )
    # -----------------------------
# Dark Mode Toggle
# -----------------------------

dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    st.markdown("""
    <style>

    .stApp{
        background-color:#0E1117;
        color:white;
    }

    section[data-testid="stSidebar"]{
        background:#161B22;
    }

    h1,h2,h3,h4,p,label{
        color:white !important;
    }

    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# Sidebar
# -----------------------------



st.sidebar.image(
    "https://img.icons8.com/fluency/96/money-bag-rupee.png",
    width=90
)

st.sidebar.title("🏦 AI Loan Prediction")


st.sidebar.write("---")

st.sidebar.markdown("## 👨‍💻 Developer")

st.sidebar.write("**Aman Dwivedi**")

st.sidebar.write("B.Tech AIML")

st.sidebar.write("---")

st.sidebar.markdown(
"[🐙 GitHub](https://github.com/amandwivedi7307)"
)

st.sidebar.markdown(
"[💼 LinkedIn]( https://www.linkedin.com/in/aman-kumar-dwivedi-6b4718297)"
)

st.sidebar.write("---")

st.sidebar.success("🚀 Machine Learning Project")

st.sidebar.info("""
### Tech Stack

🐍 Python

📊 Pandas

🤖 Scikit-Learn

🎨 Streamlit

📈 Matplotlib

🧠 Logistic Regression
""")
st.sidebar.metric(
    "Model Accuracy",
    f"{performance['accuracy'] * 100:.2f}%"
)

# -----------------------------
# Title
# -----------------------------

st.markdown("""
<div style="
background:linear-gradient(90deg,#2563EB,#1E40AF,#0EA5E9);
padding:30px;
border-radius:20px;
text-align:center;
box-shadow:0 8px 30px rgba(0,0,0,.25);
margin-bottom:20px;
">

<h1 style="color:white;font-size:42px;">
🏦 AI Loan Prediction Dashboard
</h1>

<p style="color:white;font-size:20px;">
Predict Loan Approval using Machine Learning
</p>

</div>
""", unsafe_allow_html=True)
col1,col2,col3,col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        "78.38%"
    )

with col2:
    st.metric(
        "Algorithm",
        "Logistic Reg."
    )

with col3:
    st.metric(
        "Dataset",
        "614 Records"
    )

with col4:
    st.metric(
        "Features",
        "13"
    )


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

# -----------------------------
# Center Predict Button
# -----------------------------

left_space, center_button, right_space = st.columns([1, 2, 1])

with center_button:
    predict = st.button(
        "🚀 Predict Loan Approval",
        use_container_width=True
    )

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

    probs = model.predict_proba(input_df)[0]

    classes = model.classes_

    approve_index = list(classes).index("Y")

    approval_probability = probs[approve_index] * 100

    rejection_probability = 100 - approval_probability

    st.write("")
    st.write("Prediction:", prediction)
    st.write("Approval Probability:", f"{approval_probability:.2f}%")

    # -----------------------------
    # Result
    # -----------------------------

    if prediction == "Y":

        st.balloons()
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#16a34a,#22c55e);
        padding:25px;
        border-radius:18px;
        color:white;
        text-align:center;
        box-shadow:0 8px 20px rgba(0,0,0,.25);
        ">

        <h1>✅ LOAN APPROVED</h1>

        <h2>Approval Probability : {approval_probability:.2f}%</h2>

        <p>Your application has a high chance of approval.</p>

        </div>
        """, unsafe_allow_html=True)

        st.progress(approval_probability/100)

        

        

    else:
        st.snow()

        st.toast("Loan Rejected")
        st.markdown(f"""
        <div style="
        background:linear-gradient(135deg,#dc2626,#ef4444);
        padding:25px;
        border-radius:18px;
        color:white;
        text-align:center;
        box-shadow:0 8px 20px rgba(0,0,0,.25);
        ">

        <h1>❌ LOAN REJECTED</h1>

        <h2>Approval Probability : {approval_probability:.2f}%</h2>

        <p>The application has a low probability of approval.</p>

        </div>
        """, unsafe_allow_html=True)

        st.progress(approval_probability/100)

        
    pdf_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self Employed": self_emp,
            "Applicant Income": applicant_income,
            "Coapplicant Income": coapplicant_income,
            "Loan Amount": loan_amount,
            "Loan Term": loan_term,
            "Credit History": credit_history,
            "Property Area": property_area
        }

    pdf_name = create_pdf(
            pdf_data,
            "Approved" if prediction == "Y" else "Rejected",
            approval_probability
        )

    with open(pdf_name, "rb") as pdf_file:
        st.download_button(
            label="📄 Download Prediction Report",
            data=pdf_file,
            file_name=pdf_name,
            mime="application/pdf"
        )
    # gauge chart for approval probability
    fig = go.Figure(go.Indicator(

        mode="gauge+number",

        value=approval_probability,

        title={'text':"Approval Probability"},

        gauge={

            'axis':{'range':[0,100]},

            'bar':{'color':"#2563EB"},

            'steps':[

                {'range':[0,50],'color':"red"},

                {'range':[50,80],'color':"orange"},

                {'range':[80,100],'color':"green"}

            ]

        }

    ))

    st.plotly_chart(fig,use_container_width=True)
    csv=input_df.copy()

    csv["Prediction"]=prediction

    st.download_button(

    "📥 Download CSV",

    csv.to_csv(index=False),

    "Prediction.csv",

    "text/csv"

    )

    # -----------------------------
    # Metrics
    # -----------------------------

   # -----------------------------
# Prediction Summary
# -----------------------------

    st.write("")

    card1, card2, card3, card4 = st.columns(4)

    with card1:
        st.metric(
            label="🎯 Accuracy",
            value="78.38%"
        )

    with card2:
        st.metric(
            label="📈 Approval",
            value=f"{approval_probability:.2f}%"
        )

    with card3:
        st.metric(
            label="💰 Income",
            value=f"₹ {total_income:,.0f}"
        )

    with card4:
        st.metric(
            label="🏦 Loan",
            value=f"₹ {loan_amount:,.0f}"
        )
        

    # -----------------------------
    # Charts
    # -----------------------------

    st.write("---")

    c1, c2 = st.columns(2)

    with c1:

        st.subheader("Applicant Income")

        st.pyplot(
    income_chart(
        applicant_income,
        coapplicant_income
    )
)

    with c2:

        st.subheader("Loan Amount")

        st.pyplot(
    loan_chart(
        loan_amount
    )
)
        
    st.write("---")
    st.subheader("🥧 Approval Probability")

    approve = approval_probability
    reject = rejection_probability

    fig3, ax3 = plt.subplots(figsize=(5,4))

    ax3.pie(
        [approve, reject],
        labels=["Approval", "Rejection"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax3.axis("equal")

    st.pyplot(fig3)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"💰 Total Income\n\n₹ {total_income:,.0f}")

    with col2:
        st.info(f"🏦 Loan Amount\n\n₹ {loan_amount:,.0f}")

    with col3:
        st.info(f"📈 Approval Probability\n\n{approval_probability:.2f}%")

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
    st.write("---")
    st.header("📊 Financial Analysis Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Income Distribution")

        fig, ax = plt.subplots(figsize=(5,4))

        ax.bar(
            ["Applicant", "Co-Applicant"],
            [applicant_income, coapplicant_income],
            color=["#2563EB", "#16A34A"]
        )

        ax.set_ylabel("Income")

        for i, value in enumerate([applicant_income, coapplicant_income]):
            ax.text(i, value, str(value), ha="center")

        st.pyplot(fig)

    with col2:
        st.subheader("Loan vs Total Income")

        fig2, ax2 = plt.subplots(figsize=(5,4))

        ax2.bar(
            ["Total Income", "Loan Amount"],
            [total_income, loan_amount],
            color=["#0EA5E9", "#EF4444"]
        )

        for i, value in enumerate([total_income, loan_amount]):
            ax2.text(i, value, str(value), ha="center")

        st.pyplot(fig2)

# -----------------------------
# Footer
# -----------------------------

st.write("---")

st.header("📈 Model Performance")

report = performance["classification_report"]


col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Accuracy",
        f"{performance['accuracy']*100:.2f}%"
    )

    st.metric(
        "Precision",
        f"{report['Y']['precision']:.2f}"
    )

with col2:
    st.metric(
        "Recall",
        f"{report['Y']['recall']:.2f}"
    )

    st.metric(
        "F1 Score",
        f"{report['Y']['f1-score']:.2f}"
    )
st.subheader("📊 Confusion Matrix")

cm = performance["confusion_matrix"]

fig, ax = plt.subplots(figsize=(5,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

st.write("---")

st.markdown("""

<div style="text-align:center">

<h2>🏦 AI Loan Prediction Dashboard</h2>

<p>Developed by <b>Aman Dwivedi</b></p>

<p>

<a href="https://github.com/amandwivedi7307">

GitHub Repository

</a>

</p>

<p>

© 2026 All Rights Reserved

</p>

</div>

""",unsafe_allow_html=True)