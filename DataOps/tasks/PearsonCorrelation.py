import pandas as pd
from scipy.stats import pearsonr
import logging
from matplotlib import pyplot as plt
import io
import base64
import os
import re
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset
df = pd.read_csv("../data/train.csv")
logger.info("Dataset loaded for Pearson correlation calculation on Outstanding_Debt and Credit Score.")


# Clean and convert 'Annual_Income' to numeric by removing non-numeric characters
df['Annual_Income'] = df['Annual_Income'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x)))
df['Annual_Income'] = pd.to_numeric(df['Annual_Income'], errors='coerce')

# Annual Income Binning
income_labels = ['Low Income', 'Medium Income', 'High Income', 'Very High Income']
df['Income_Binned'] = pd.qcut(df['Annual_Income'], q=4, labels=income_labels, precision=1)
logger.info(f"Annual Income Binning:\n{df['Income_Binned'].value_counts()}")

# Ensure the output directory exists
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

# Plot Credit Score distribution by Income Binned
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Income_Binned', hue='Credit_Score', palette="viridis")
plt.title('Distribution of Credit Score by Annual Income Bins')
plt.xlabel('Income Binned')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Credit Score')

# Save the plot to an output file
output_file = os.path.join(output_dir, "credit_score_by_income_bins.png")
plt.savefig(output_file)
logger.info(f"Credit Score by Annual Income Binned bar plot saved as '{output_file}'")
plt.close()

# Convert 'Outstanding_Debt' and 'Credit_Score' to numeric, coercing errors
df['Outstanding_Debt'] = df['Outstanding_Debt'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x)))
df['Outstanding_Debt'] = pd.to_numeric(df['Outstanding_Debt'], errors='coerce')


# Remove rows where Outstanding_Debt is NaN, negative, or greater than 100, and where Credit_Score is NaN

# Convert relevant columns to series
credit_score_mapping = {'Good': 3, 'Standard': 2, 'Poor': 1}
df['Credit_Score_Numeric'] = df['Credit_Score'].map(credit_score_mapping)

list1 = df['Outstanding_Debt']
list2 = df['Credit_Score_Numeric']
logger.info("list1")
logger.info(list1)
logger.info("list2")
logger.info(list2)

# Compute Pearson correlation
corr, _ = pearsonr(list1, list2)
logger.info(f'Pearson correlation between Outstanding_Debt and Credit Score: {corr:.3f}')

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(list1, list2, alpha=0.5)
plt.xlabel('Outstanding_Debt')
plt.ylabel('Credit Score')
plt.title('Scatter plot of Outstanding_Debt vs Credit Score')

# Save the plot to a buffer
buf = io.BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)

# Encode the imOutstanding_Debt in base64 and log it
img_base64 = base64.b64encode(buf.read()).decode('utf-8')
#logger.info("Scatter plot imOutstanding_Debt (base64 encoded):")
#logger.info(f"data:imOutstanding_Debt/png;base64,{img_base64}")

# Ensure the output directory exists
output_dir = "../output"
os.makedirs(output_dir, exist_ok=True)

# Save the plot as a file
output_file = os.path.join(output_dir, "Outstanding_Debt_vs_credit_score_scatter_plot.png")
plt.savefig(output_file)
logger.info(f"Scatter plot saved as '{output_file}'")

# Close the buffer
buf.close()
plt.close()
