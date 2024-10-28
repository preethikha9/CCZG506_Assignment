import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Configure standard logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the dataset
df = pd.read_csv("../data/train.csv",low_memory=False)
logger.info("Dataset loaded successfully.")
logger.info(f"DataFrame head:\n{df.head()}")

# Summary statistics
logger.info("Summary Statistics:")
logger.info(f"\n{df.describe()}")

# Checking for missing values
logger.info("Missing Values:")
logger.info(f"\n{df.isnull().sum()}")

# Data type information
logger.info("Data Types:")
logger.info(f"\n{df.dtypes}")

#Distribution of data 
output_dir = "../output"
logger.info("Distribution of data:")
plt.figure(figsize=(20, 15))
df.hist(figsize=(20, 15))
output_file = os.path.join(output_dir, "all_histograms.png")
plt.savefig(output_file)


# Look at each column
logger.info("Values at each column:")
for column in df.columns:
    logger.info(f"\n{column}:\n{df[column].value_counts()}")
