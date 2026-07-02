import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# LOAD DATASET
df = pd.read_csv("loan_approval_dataset.csv")
df.head() # display the first 5 rows of the dataset

# Basic checks
print(df.shape) # check the shape of the dataset
print(df.info) # check the data types and missing values
print(df.isnull().sum()) # check for missing values in each column

# Data Cleaning (fill missing values)

df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median()) # fill missing values in LoanAmount column with median
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0]) # fill missing values in Loan_Amount_Term column with mode
df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0]) # fill missing values in Credit_History column with mode
categorical_cols = ['Gender' , 'Married' , 'Dependents' , 'Education'  , 'Self_Employed'] # list of categorical columns in the dataset
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0]) # fill missing values in categorical columns with mode

# Exploratory Data Analysis (EDA)  [credit history vs loan Status]

sns.countplot(x = 'Credit_History', hue = 'Loan_Status' , data = df) # this will show the count of loan status based on credit history in the dataset in graphical format
plt.show()
# Income Distribution
sns.histplot(df['ApplicantIncome'], kde=True) # this will show the distribution of applicant income in the dataset in graphical format
plt.show()

# Feature Engineering (Encoding categorical variables) [create total income column]

df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome'] # create a new column 'Total_Income' by adding 'ApplicantIncome' and 'CoapplicantIncome'
df['LoanAmount_log'] = np.log(df['LoanAmount']) # log tranform of 'LoanAmount' column to reduce skewness

#Encode categorical variables using LabelEncoder

le = LabelEncoder()
for col in df.select_dtypes(include=['object' , 'string']).columns: # loop through all categorical columns in the dataset
    df[col] = le.fit_transform(df[col]) # encode categorical columns using LabelEncoder

# split features and target variable

X = df.drop("Loan_Status" , axis = 1) # drop the target variable 'Loan_Status' from the dataset to create feature set X
Y = df["Loan_Status"] # create target variable Y by selecting 'Loan_Status' column from the dataset
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.3 , random_state = 42) # split the dataset into training and testing sets with 70% training and 30% testing and random state of 42 for reproducibility

# Build Model (Logistic Regression)

model = LogisticRegression(max_iter=10000) # create an instance of LogisticRegression model
model.fit(X_train , Y_train) # fit the model on the training data

#Prediction

Y_pred = model.predict(X_test) # make predictions on the test data

# Evaluation

accuracy = accuracy_score(Y_test , Y_pred) # calculate accuracy of the model
print("Accuracy of the model:" , accuracy * 100, "%") # print the accuracy of the model
print("Confusion Matrix:")
print(confusion_matrix(Y_test , Y_pred)) # print the confusion matrix of the model

print(classification_report(Y_test , Y_pred)) # print the classification report of the model






