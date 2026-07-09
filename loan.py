import pandas as pd
import numpy as np
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("loan_approval_dataset.csv")

# Feature Engineering
df["Total_Income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
df["LoanAmount_log"] = np.log(df["LoanAmount"].fillna(df["LoanAmount"].median()))

# Remove Loan_ID
df.drop("Loan_ID", axis=1, inplace=True)

# Split
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

categorical_features = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area"
]

numerical_features = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Total_Income",
    "LoanAmount_log"
]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=5000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, pred))

joblib.dump(model, "loan_model.pkl")

print("Model Saved Successfully!")
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import joblib

performance = {
    "accuracy": accuracy_score(y_test, pred),
    "confusion_matrix": confusion_matrix(y_test, pred),
    "classification_report": classification_report(
        y_test,
        pred,
        output_dict=True
    )
}

joblib.dump(model, "loan_model.pkl")
joblib.dump(performance, "performance.pkl")

print("Model and performance saved successfully!")