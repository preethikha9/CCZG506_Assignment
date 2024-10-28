import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report,accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import psutil
import time
 
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load data
def load_data():
    train = pd.read_csv("train.csv", low_memory=False)
    return train

# Filter columns based on certain conditions
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

def preprocess_data(train):
    # Drop unnecessary columns
    train.drop(["ID", "Customer_ID", "Name", "SSN", "Month", "Type_of_Loan"], axis=1, inplace=True)
    
    # Replace specific values in columns
    train['Credit_Mix'] = train['Credit_Mix'].replace('_', np.nan)
    train['Changed_Credit_Limit'] = train['Changed_Credit_Limit'].replace('_', np.nan).astype("float")
    train['Changed_Credit_Limit'] = train['Changed_Credit_Limit'].fillna(train["Changed_Credit_Limit"].mean()).round(3)
    train['Monthly_Balance'] = train['Monthly_Balance'].replace('__-333333333333333333333333333__', np.nan).astype("float")
    
    # Replace strings in Payment_Behaviour and map to integers
    train["Payment_Behaviour"] = train["Payment_Behaviour"].replace({
        "!@9#%8": np.nan, "Low_spent_Small_value_payments": 1, "Low_spent_Medium_value_payments": 2, 
        "Low_spent_Large_value_payments": 3, "High_spent_Small_value_payments": 4, 
        "High_spent_Medium_value_payments": 5, "High_spent_Large_value_payments": 6
    })
    
    # Apply custom filters
    train["Outstanding_Debt"] = train["Outstanding_Debt"].apply(filter_col).astype(float)
    train["Age"] = train["Age"].apply(filter_col).astype(int)
    train.loc[(train["Age"] > 90) | (train["Age"] < 10), "Age"] = np.nan
    train["Annual_Income"] = train["Annual_Income"].apply(filter_col).astype(float)
    train["Num_of_Loan"] = train["Num_of_Loan"].apply(filter_col).astype(int)
    train.loc[train["Num_of_Loan"] > 100, "Num_of_Loan"] = np.nan
    train["Occupation"] = train["Occupation"].replace("_______", np.nan).astype("object")
    train["Num_of_Delayed_Payment"] = train["Num_of_Delayed_Payment"].apply(filter_).replace("NaN", np.nan).astype("float")
    
    # Replace '__10000__' in Amount_invested_monthly with NaN and convert to float
    train["Amount_invested_monthly"] = train["Amount_invested_monthly"].replace(["NaN", "__10000__"], np.nan).astype("float").round(3)
    
    # Split credit history into years and months
    years, months = split_credit_history(train)
    train['Credit_Age_years'] = pd.Series(years)
    train['Credit_Age_months'] = pd.Series(months)
    train.drop('Credit_History_Age', axis=1, inplace=True)

    # Additional data cleaning and formatting
    train["Delay_from_due_date"] = train["Delay_from_due_date"].clip(lower=0)
    train["Num_Bank_Accounts"] = train["Num_Bank_Accounts"].replace(-1, 0)
    train["Credit_Utilization_Ratio"] = train["Credit_Utilization_Ratio"].round(2)
    train["Total_EMI_per_month"] = train["Total_EMI_per_month"].astype("float").round(3)
    
    # Encode categorical data
    train['Credit_Mix'] = train['Credit_Mix'].replace({"Standard": 1, "Bad": 2, "Good": 3, "-": np.nan})
    return train


# Split Credit History into years and months
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

# Impute missing values
def impute_missing_values(train):
    numerical_data = [col for col in train.columns if train[col].dtype != 'object']
    knn_imputer = KNNImputer(n_neighbors=5)
    simple_imputer = SimpleImputer(strategy="mean")
    train[numerical_data] = simple_imputer.fit_transform(train[numerical_data])
    train['Credit_Mix'] = knn_imputer.fit_transform(train['Credit_Mix'].values.reshape(-1, 1))
    
    # Forward fill categorical missing values
    train["Payment_Behaviour"].ffill(inplace=True)
    train['Occupation'].ffill(inplace=True)
    
    # Encode categorical columns
    encode_columns(train)
    return train

# Encode categorical columns
def encode_columns(train):
    columns = ['Credit_Score', 'Occupation', 'Payment_of_Min_Amount']
    for item in columns:
        train[item] = LabelEncoder().fit_transform(train[item])

# Train and evaluate the model
def train_model(X_train, y_train):
    model = RandomForestClassifier( random_state=1234)
    # Train the model on the training set
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy on the test set
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy on test set:", accuracy)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def log_to_mlflow(model, X_train, X_test, y_train, y_test):
    with mlflow.start_run():
        mlflow.set_tag("Created by", "Team36")
        # Log hyper parameters using in Random Forest Algorithm
        # mlflow.log_param("max_depth", model.max_depth)
        # mlflow.log_param("n_estimators", model.n_estimators)

        # Log model metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='micro')
        recall = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='micro')
        confusion = confusion_matrix(y_test, y_pred)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1-score", f1)
        
        # Log confusion matrix
        confusion_dict = {
            "true_positive": confusion[1][1],
            "false_positive": confusion[0][1],
            "true_negative": confusion[0][0],
            "false_negative": confusion[1][0]
        }
        mlflow.log_metrics(confusion_dict)

        # Log system metrics
        # Example: CPU and Memory Usage
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_usage = psutil.virtual_memory().percent

        mlflow.log_metric("system_cpu_usage", cpu_usage)
        mlflow.log_metric("system_memory_usage", memory_usage)

        # Log execution time for training the model
        execution_time = {}  # Dictionary to store execution times for different stages
        # Example: Execution time for training the model
        start_time = time.time()
        model = train_model(X_train, y_train)
        end_time = time.time()
        execution_time["system_model_training"] = end_time - start_time

        # Log execution time 
        mlflow.log_metrics(execution_time)

        # Evaluate model and log metrics
        evaluate_model(model, X_test, y_test)

        # Log model
        mlflow.sklearn.log_model(model, "model")

# Main function to run all steps
def main():
    train = load_data()
    train = preprocess_data(train)
    train = impute_missing_values(train)
    X = train.drop("Credit_Score", axis=1)
    y = train["Credit_Score"]
    # Split into training and testing sets (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
    model = train_model(X_train, y_train)
    log_to_mlflow(model, X_train, X_test, y_train, y_test)

# Execute main function
if __name__ == "__main__":
    main()
