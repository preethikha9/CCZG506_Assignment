import pandas as pd
import numpy as np
import logging
import re  # Import re for regex cleaning

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset
df = pd.read_csv("../data/train.csv")
logger.info("Dataset loaded for binning.")

# Clean and convert 'Annual_Income' to numeric by removing non-numeric characters
df['Annual_Income'] = df['Annual_Income'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x)))
df['Annual_Income'] = pd.to_numeric(df['Annual_Income'], errors='coerce')

# Annual Income Binning
income_labels = ['Low Income', 'Medium Income', 'High Income', 'Very High Income']
df['Income_Binned'] = pd.qcut(df['Annual_Income'], q=4, labels=income_labels, precision=1)
logger.info(f"Annual Income Binning:\n{df['Income_Binned'].value_counts()}")

# Clean and convert 'Outstanding_Debt' to numeric
df['Outstanding_Debt'] = df['Outstanding_Debt'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x)))
df['Outstanding_Debt'] = pd.to_numeric(df['Outstanding_Debt'], errors='coerce')

# Outstanding Debt Binning
debt_labels = ['Low Debt', 'Moderate Debt', 'High Debt', 'Very High Debt']
df['Debt_Binned'] = pd.qcut(df['Outstanding_Debt'], q=4, labels=debt_labels, precision=1)
logger.info(f"Outstanding Debt Binning:\n{df['Debt_Binned'].value_counts()}")

# Extract numeric years from 'Credit_History_Age' for binning
df['Credit_History_Age'] = df['Credit_History_Age'].str.extract(r'(\d+)')[0].astype(float)

# Credit History Age Binning
credit_age_bins = [0, 5, 10, 15, 20, 30]
credit_age_labels = ['New', 'Moderate', 'Established', 'Mature', 'Veteran']
df['Credit_History_Binned'] = pd.cut(df['Credit_History_Age'], bins=credit_age_bins, labels=credit_age_labels, include_lowest=True)
logger.info(f"Credit History Age Binning:\n{df['Credit_History_Binned'].value_counts()}")

# Total EMI Per Month Binning
emi_labels = ['Low EMI', 'Medium EMI', 'High EMI', 'Very High EMI']
df['EMI_Binned'] = pd.qcut(df['Total_EMI_per_month'], q=4, labels=emi_labels, precision=1)
logger.info(f"Total EMI Per Month Binning:\n{df['EMI_Binned'].value_counts()}")

# Number of Bank Accounts Binning
bank_accounts_bins = [0, 2, 4, 6, 10, df['Num_Bank_Accounts'].max()]
bank_accounts_labels = ['Very Few', 'Few', 'Moderate', 'Many', 'Too Many']
df['Bank_Accounts_Binned'] = pd.cut(df['Num_Bank_Accounts'], bins=bank_accounts_bins, labels=bank_accounts_labels, include_lowest=True)
logger.info(f"Number of Bank Accounts Binning:\n{df['Bank_Accounts_Binned'].value_counts()}")

#encoding
df['Credit_Mix'] = df['Credit_Mix'].replace({"Standard": 1, "Bad": 2, "Good": 3, "-": np.nan})

# Number of Credit Cards Binning
credit_card_bins = [0, 2, 4, 6, 10, df['Num_Credit_Card'].max()]
credit_card_labels = ['Very Few', 'Few', 'Moderate', 'Many', 'Too Many']
df['Credit_Card_Binned'] = pd.cut(df['Num_Credit_Card'], bins=credit_card_bins, labels=credit_card_labels, include_lowest=True)
logger.info(f"Number of Credit Cards Binning:\n{df['Credit_Card_Binned'].value_counts()}")
