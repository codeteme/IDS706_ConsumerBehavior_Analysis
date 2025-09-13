import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor
from sklearn import tree

from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

plt.style.use("fivethirtyeight")

def load_data(file_url) -> pd.DataFrame: 
    data = pd.read_csv(file_url)
    return data

def basic_data_analysis(data: pd.DataFrame): 
    data.head()
    data.info()
    data.describe(include="all")
    return 

def clean_data(data: pd.DataFrame) -> pd.DataFrame:  
    # Data Cleaning
    # Looking at data.info() and checking against data, 'Purchase_Amount' could be converted to a float
    data["Purchase_Amount"] = data["Purchase_Amount"].str.replace("$", "").astype(float)
    assert data["Purchase_Amount"].dtype == "float64"

    z_scores = np.abs(stats.zscore(data.select_dtypes(include=np.number)))
    print(f"There are a total of {data[(z_scores > 3).all(axis = 1)].shape[0]} outliers")
    print(f"There are a total of {data.duplicated().sum()} duplicated row(s)")

    for col in data.columns:
        if data[col].isnull().sum() > 0:
            print(f"{col} has {data[col].isnull().sum()} null values")

    print("---Handling missing values---")
    data["Social_Media_Influence"] = data["Social_Media_Influence"].fillna(
        data["Social_Media_Influence"].mode()[0]
    )
    data["Engagement_with_Ads"] = data["Engagement_with_Ads"].fillna(
        data["Engagement_with_Ads"].mode()[0]
    )
    print(
        f"Total number of columns with missing values after clean up: {data.isnull().any().sum()}"
    )
    return data

def exploratory_data_analysis(data: pd.DataFrame) -> list[str]:
    # Exploratory Data Analysis (EDA)
    data.groupby("Income_Level")["Purchase_Amount"].mean()

    # Distribution of numerical features
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(["Age", "Purchase_Amount"]):
        plt.subplot(1, 2, i + 1)
        sns.histplot(x=data[col], kde=True)
        plt.title(f"Distribution of {col}")
    plt.tight_layout()
    # plt.show() # Uncomment to display figure

    # Categorical feature counts
    string_cols = data.select_dtypes(include="object").columns.tolist()
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(["Gender", "Marital_Status", "Income_Level"]):
        plt.subplot(1, 3, i + 1)
        sns.countplot(x=data[col])
        plt.title(f"Count of {col}")
    plt.tight_layout()
    # plt.show() # Uncomment to display figure

    # Correlation Heatmap for numerical features
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix of Numeric Features")
    # plt.show() # Uncomment to display figure

    return string_cols

def data_transformation_feature_engineering(data: pd.DataFrame, string_cols: list) -> pd.DataFrame:
    cols_to_onehot = [c for c in string_cols if c not in ["Customer_ID", "Time_of_Purchase", "Location"]]
    data_encoded = pd.get_dummies(data, columns=cols_to_onehot, drop_first=False)
    data_final = data_encoded.drop(["Customer_ID", "Location", "Time_of_Purchase"], axis=1)
    return data_final

def regression_analysis(data: pd.DataFrame): 
    # Regression Analysis: Predicting Purchase_Amount
    X_reg = data.drop("Purchase_Amount", axis=1)
    y_reg = data["Purchase_Amount"]

    # Scale numerical features (excluding the target)
    scaler = StandardScaler()
    X_reg_scaled = X_reg.copy()
    numeric_features_reg = X_reg_scaled.select_dtypes(include=np.number).columns.tolist()
    X_reg_scaled[numeric_features_reg] = scaler.fit_transform(X_reg_scaled[numeric_features_reg])

    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_scaled, y_reg, test_size=0.2, random_state=42
    )
    print("Data split into training and testing sets for regression.")

    # Linear Regression
    reg = LinearRegression().fit(X_train_reg, y_train_reg)
    y_pred_reg_lin = reg.predict(X_test_reg)
    mse_lin = mean_squared_error(y_test_reg, y_pred_reg_lin)
    rmse_lin = np.sqrt(mse_lin)
    print(f"Linear Regression | MSE: {mse_lin:.2f}, RMSE: {rmse_lin:.2f}")

    # Feature Importance plot
    importance = reg.coef_
    features = reg.feature_names_in_
    plt.figure(figsize=(8, 5))
    plt.barh(features, importance)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Importance")
    # plt.show() # Uncomment to display figure

    # XGBoost Regression
    xgbreg = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    xgbreg.fit(X_train_reg, y_train_reg)
    y_pred_reg_xgb = xgbreg.predict(X_test_reg)
    mse_xgb = mean_squared_error(y_test_reg, y_pred_reg_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    print(f"XGBoost Regression | MSE: {mse_xgb:.2f}, RMSE: {rmse_xgb:.2f}")

    # XGBoost feature importance plot
    xgb_importance = xgbreg.feature_importances_
    xgb_features = X_train_reg.columns
    plt.figure(figsize=(15, 10))
    plt.barh(xgb_features, xgb_importance)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature Importance")
    # plt.show() # Uncomment to display figure

    return


def classification_analysis(data: pd.DataFrame): 
    # Classification Analysis: Predicting Customer_Satisfaction
    X_clf = data.drop("Customer_Satisfaction", axis=1)
    y_clf = data["Customer_Satisfaction"].astype("category")
    print("Defined features and target for classification.")

    # Scale numerical features
    scaler_clf = StandardScaler()
    X_clf_scaled = X_clf.copy()
    numeric_features_clf = X_clf.select_dtypes(include=np.number).columns.tolist()
    X_clf_scaled[numeric_features_clf] = scaler_clf.fit_transform(X_clf_scaled[numeric_features_clf])

    # Split data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf_scaled, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    print("Data split into training and testing sets for classification.")

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_clf, y_train_clf)
    y_pred_clf_log = logreg.predict(X_test_clf)
    acc_log = accuracy_score(y_test_clf, y_pred_clf_log)
    print("\nLogistic Regression:")
    print(f"Accuracy: {acc_log:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_clf, y_pred_clf_log))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_clf, y_pred_clf_log))

    # Decision Tree Classifier
    clf_tree = tree.DecisionTreeClassifier(random_state=42)
    clf_tree.fit(X_train_clf, y_train_clf)
    y_pred_clf_tree = clf_tree.predict(X_test_clf)
    acc_tree = accuracy_score(y_test_clf, y_pred_clf_tree)
    print("\nDecision Tree Classifier:")
    print(f"Accuracy: {acc_tree:.4f}")
    print("Classification Report:")
    print(classification_report(y_test_clf, y_pred_clf_tree))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_clf, y_pred_clf_tree))


if __name__ == "__main__": 
    file_path = "https://raw.githubusercontent.com/codeteme/IDS706_DE_Wk2/refs/heads/main/data/raw/Ecommerce_Consumer_Behavior_Analysis_Data.csv"
    data = load_data(file_path)
    basic_data_analysis(data)
    data = clean_data(data)
    string_cols = exploratory_data_analysis(data)
    data_final = data_transformation_feature_engineering(data, string_cols)
    regression_analysis(data_final)
    classification_analysis(data_final)











