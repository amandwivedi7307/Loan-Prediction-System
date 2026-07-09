import matplotlib.pyplot as plt


def income_chart(applicant_income, coapplicant_income):

    fig, ax = plt.subplots(figsize=(5,3))

    ax.bar(
        ["Applicant", "Co-Applicant"],
        [applicant_income, coapplicant_income]
    )

    ax.set_title("Income Comparison")

    ax.set_ylabel("Income")

    return fig


def loan_chart(loan_amount):

    fig, ax = plt.subplots(figsize=(5,3))

    ax.bar(
        ["Loan Amount"],
        [loan_amount]
    )

    ax.set_title("Requested Loan")

    ax.set_ylabel("Amount")

    return fig


def probability_color(probability):

    if probability >= 80:
        return "green"

    elif probability >= 50:
        return "orange"

    return "red"