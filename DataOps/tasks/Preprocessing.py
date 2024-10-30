# Drop unnecessary columns
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

df = pd.read_csv("../data/train.csv")
logger.info("Dataset loaded for Pearson correlation calculation on Outstanding_Debt and Credit Score.")

def split_credit_history(train):
    years = []
    months = []
    for value in train["Credit_History_Age"]:
        if value is np.nan:
            years.append(np.nan)
            months.append(np.nan)
        else:
            new_str = value.lower().split()
            years_ = int(new_str[0])
            months_ = int(new_str[new_str.index('and') + 1])
            years.append(years_)
            months.append(months_)
    return years, months

def filter_col(value):
    if '-' in value:
        return value.split('-')[1]
    elif '_' in value:
        return value.split('_')[0]
    else:
        return value

def filter_(value: str):
    if '_' in str(value):
        return value.split('_')[0]
    else:
        return value
    

df.drop(["ID", "Customer_ID", "Name", "SSN", "Month", "Type_of_Loan"], axis=1, inplace=True)


# Replace specific values in columns
df['Credit_Mix'] = df['Credit_Mix'].replace('_', np.nan)
df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].replace('_', np.nan).astype("float")
df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].fillna(df["Changed_Credit_Limit"].mean()).round(3)
df['Monthly_Balance'] = df['Monthly_Balance'].replace('__-333333333333333333333333333__', np.nan).astype("float")

# Replace strings in Payment_Behaviour and map to integers
df["Payment_Behaviour"] = df["Payment_Behaviour"].replace({
    "!@9#%8": np.nan, "Low_spent_Small_value_payments": 1, "Low_spent_Medium_value_payments": 2, 
    "Low_spent_Large_value_payments": 3, "High_spent_Small_value_payments": 4, 
    "High_spent_Medium_value_payments": 5, "High_spent_Large_value_payments": 6
})

# Apply custom filters
df["Outstanding_Debt"] = df["Outstanding_Debt"].apply(filter_col).astype(float)
df["Age"] = df["Age"].apply(filter_col).astype(int)
df.loc[(df["Age"] > 90) | (df["Age"] < 10), "Age"] = np.nan
df["Annual_Income"] = df["Annual_Income"].apply(filter_col).astype(float)
df["Num_of_Loan"] = df["Num_of_Loan"].apply(filter_col).astype(int)
df.loc[df["Num_of_Loan"] > 100, "Num_of_Loan"] = np.nan
df["Occupation"] = df["Occupation"].replace("_______", np.nan).astype("object")
df["Num_of_Delayed_Payment"] = df["Num_of_Delayed_Payment"].apply(filter_).replace("NaN", np.nan).astype("float")

# Replace '__10000__' in Amount_invested_monthly with NaN and convert to float
df["Amount_invested_monthly"] = df["Amount_invested_monthly"].replace(["NaN", "__10000__"], np.nan).astype("float").round(3)

# Split credit history into years and months
years, months = split_credit_history(df)
df['Credit_Age_years'] = pd.Series(years)
df['Credit_Age_months'] = pd.Series(months)
df.drop('Credit_History_Age', axis=1, inplace=True)

# Additional data cleaning and formatting
df["Delay_from_due_date"] = df["Delay_from_due_date"].clip(lower=0)
df["Num_Bank_Accounts"] = df["Num_Bank_Accounts"].replace(-1, 0)
df["Credit_Utilization_Ratio"] = df["Credit_Utilization_Ratio"].round(2)
df["Total_EMI_per_month"] = df["Total_EMI_per_month"].astype("float").round(3)

# Encode categorical data
df['Credit_Mix'] = df['Credit_Mix'].replace({"Standard": 1, "Bad": 2, "Good": 3, "-": np.nan})
df.to_csv('../../MLOps/train.csv', index=False)

logger.info(df.head(10))