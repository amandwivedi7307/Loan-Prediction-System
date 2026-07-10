import numpy as np
import pandas as pd


def prepare_input(
    gender,
    married,
    dependents,
    education,
    self_emp,
    applicant_income,
    coapplicant_income,
    loan_amount,
    loan_term,
    credit_history,
    property_area,
):

    total_income = applicant_income + coapplicant_income

    loan_log = np.log(loan_amount)

    input_df = pd.DataFrame({

        "Gender":[gender],

        "Married":[married],

        "Dependents":[dependents],

        "Education":[education],

        "Self_Employed":[self_emp],

        "ApplicantIncome":[applicant_income],

        "CoapplicantIncome":[coapplicant_income],

        "LoanAmount":[loan_amount],

        "Loan_Amount_Term":[loan_term],

        "Credit_History":[credit_history],

        "Property_Area":[property_area],

        "Total_Income":[total_income],

        "LoanAmount_log":[loan_log]

    })

    return input_df


def predict_loan(model,input_df):

    prediction=model.predict(input_df)[0]

    probs=model.predict_proba(input_df)[0]

    classes=model.classes_

    approve_index=list(classes).index("Y")

    approval_probability=probs[approve_index]*100

    rejection_probability=100-approval_probability

    return (

        prediction,

        approval_probability,

        rejection_probability

    )