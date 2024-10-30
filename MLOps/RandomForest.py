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
    # train = preprocess_data(train)
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
