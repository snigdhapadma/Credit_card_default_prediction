# 💳 Credit Card Default Risk Prediction

## Project Overview

Credit card default is an important risk faced by financial institutions when customers fail to repay their credit card dues on time.This project develops a Machine Learning classification system to predict whether a credit card customer is likely to default based on demographic information, credit limits, repayment history, bill amounts, and payment amounts.

## Objective

The main objective of this project is to analyze historical credit card customer data and develop a machine learning classification model that can predict credit card default risk.
The system can help financial institutions:

- Identify customers who are at higher risk of default
- Understand repayment behavior
- Improve credit risk assessment
- Support data-driven lending decisions
- Reduce potential financial losses
  
## 📊 Dataset

The dataset contains:

- 30,000 customer records
- 25 columns
- 1 target variable

The dataset contains information related to:

**Customer Demographics**

Gender (SEX),Education (EDUCATION),Marriage status (MARRIAGE),Age (AGE)

**Credit Information**

Credit limit (LIMIT_BAL)

**Repayment History**

PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6

**Bill Amounts**

BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6

**Payment Amounts**

PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6

**target variable**

default:

0 → Customer did not default

1 → Customer defaulted

## Exploratory Data Analysis

Key analyses of EDA includes:

- Default rate by education
- Default distribution across age groups
- Credit limit versus default
- Repayment behavior versus default
- Bill amounts versus default
- Payment amounts versus default
- Demographic characteristics of defaulting customers

**Important Finding**

Repayment history was found to have a strong association with default.

## Data Preprocessing

The following preprocessing steps were performed:

- Removed unwanted spaces from column names.
- Removed the customer ID from the modeling features.
- Converted SEX into numerical values.
- Cleaned EDUCATION categories.
- Cleaned MARRIAGE categories.
- Created age groups for analysis.
- Applied one-hot encoding to categorical variables.
- Converted the target variable into binary numerical form.
- Checked for missing values and duplicate records.
- Removed AGE_GROUP before model training.

## Feature Engineering

Several additional features were created to capture customer financial and repayment behavior.

**Bill-related Features**

TOTAL_BILL,AVG_BILL,MAX_BILL

**Payment-related Features**

TOTAL_PAYMENT,AVG_PAYMENT,PAYMENT_STD,ZERO_PAYMENT_MONTHS,Repayment-delay Features,TOTAL_PAY_DELAY,MAX_PAY_DELAY,NUM_DELAYED_MONTHS,AVG_PAY_DELAY

**Additional Financial Features**

BILL_CHANGE_1M,CREDIT_UTILIZATION,AVG_CREDIT_UTILIZATION,PAYMENT_TO_BILL_RATIO

These features help the model capture repayment patterns and financial behavior more effectively.

## Machine Learning Models building and evaluation:

Several classification algorithms were evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Random Forest
- Bagging
- Boosting
- Support Vector Machine (SVM)

The models were evaluated using multiple classification metrics:

Accuracy,Precision,Recall,F1 Score,ROC-AUC

<img width="690" height="275" alt="image" src="https://github.com/user-attachments/assets/bf4611ec-4393-4d7f-88ee-f2bbf1cbae07" />


Because the target variable is imbalanced, F1 Score and Recall are particularly important for evaluating the model's ability to identify defaulting customers.


## Final Model: Random Forest

Random Forest was selected as the deployment model after hyperparameter tuning.

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 78.43% |
| Precision | 51.10% |
| Recall    | 57.95% |
| F1 Score  | 54.31% |
| ROC-AUC   | 77.81% |


## Project Structure

credit_card_default_prediction/
│
├── app.py
├── Credit Card Defaulter Prediction.csv
├── credit_card_default_model.pkl
├── requirements.txt
└── README.md
