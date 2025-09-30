import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBRegressor, plot_importance
from sklearn import tree

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

plt.style.use("fivethirtyeight")


# Setup image saving
IMAGES_PATH = Path() / "images" / "classification"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def load_data(file_url) -> pd.DataFrame:
    return pd.read_csv(file_url)


def clean_data(data: pd.DataFrame, col_to_categorical: list[str]) -> pd.DataFrame:
    for col in col_to_categorical:
        data[col] = data[col].astype("category")
    # Looking at data.info() and checking against data, 'Purchase_Amount' could be converted to a float
    data["Purchase_Amount"] = data["Purchase_Amount"].str.replace("$", "").astype(float)
    # convert to datatime
    data["Time_of_Purchase"] = pd.to_datetime(data["Time_of_Purchase"], errors="coerce")
    # Make sure no customer id is duplicated
    assert not data.duplicated(subset="Customer_ID").any()

    z_scores = np.abs(stats.zscore(data.select_dtypes(include=np.number)))
    print(
        f"There are a total of {data[(z_scores > 3).all(axis = 1)].shape[0]} outliers"
    )
    print(f"There are a total of {data.duplicated().sum()} duplicated row(s)")

    for col in data.columns:
        if data[col].isnull().sum() > 0:
            print(f"{col} has {data[col].isnull().sum()} null values")
    print("---Handling missing values---")

    imputer = SimpleImputer(strategy="most_frequent")
    data[["Social_Media_Influence", "Engagement_with_Ads"]] = imputer.fit_transform(
        data[["Social_Media_Influence", "Engagement_with_Ads"]]
    )
    print(
        f"Total number of columns with missing values after clean up: {data.isnull().any().sum()}"
    )
    return data


def plot_hist(data, col, bins=50, save_name=None):
    plt.figure(figsize=(15, 5))
    plt.hist(data[col], bins=bins)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    if save_name:
        save_fig(save_name)
    # plt.show() # Uncomment to see figure


def plot_boxgrid(data, cols, n_cols=3, save_name=None):
    n_rows = math.ceil(len(cols) / n_cols)
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(x=data[col])
        plt.title(col)
        plt.xticks(rotation=45)
    plt.tight_layout()
    if save_name:
        save_fig(save_name)
    # plt.show() # Uncomment to see figure


def plot_countgrid(data, cols, n_cols=4, save_name=None):
    n_rows = math.ceil(len(cols) / n_cols)
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.countplot(x=data[col])
        plt.title(col)
        plt.xticks(rotation=45)
    plt.tight_layout()
    if save_name:
        save_fig(save_name)
    plt.show()


def plot_corr_heatmap(data, numerical_cols, save_name=None):
    plt.figure(figsize=(12, 8))
    sns.heatmap(data[numerical_cols].corr(), annot=True)
    plt.title("Correlation Matrix of Numeric Features")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    if save_name:
        save_fig(save_name)
    # plt.show() # Uncomment to see figure


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    # Drop irrelevant columns
    object_cols = data.select_dtypes(
        include=["object", "datetime64[ns]"]
    ).columns.tolist()
    data = data.drop(object_cols, axis=1)

    # Categorical feature counts
    string_cols = data.select_dtypes(include=["category"]).columns.tolist()

    # One-hot encode categorical features
    cols_to_onehot = data[string_cols].columns.tolist()
    data = pd.get_dummies(data, columns=cols_to_onehot, drop_first=True)
    print("Categorical columns have been one-hot encoded.")

    return data


def regression_analysis(data: pd.DataFrame):
    # Regression Analysis: Predicting Purchase_Amount
    X_reg = data.drop("Purchase_Amount", axis=1)
    y_reg = data["Purchase_Amount"]

    # Split data
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    # Scale numerical features (excluding the target)
    numeric_features_reg = X_train_reg.select_dtypes(include=np.number).columns.tolist()

    scaler = StandardScaler()
    X_train_reg[numeric_features_reg] = scaler.fit_transform(
        X_train_reg[numeric_features_reg]
    )
    X_test_reg[numeric_features_reg] = scaler.transform(
        X_test_reg[numeric_features_reg]
    )

    print("Data split into training and testing sets for regression.")

    # Linear Regression Model
    reg = LinearRegression()
    reg.fit(X_train_reg, y_train_reg)
    y_pred_reg_lin = reg.predict(X_test_reg)

    mse_lin = mean_squared_error(y_test_reg, y_pred_reg_lin)
    rmse_lin = np.sqrt(mse_lin)
    r2_lin = r2_score(y_test_reg, y_pred_reg_lin)
    print(
        f"Linear Regression | MSE: {mse_lin:.2f}, RMSE: {rmse_lin:.2f}",
        f"Linear Regression | r2: {r2_lin:.2f}, RMSE: {r2_lin:.2f}",
    )  # Linear Regression | MSE: 18819.68, RMSE: 137.18 | r2: -0.08, RMSE: -0.08

    feature_importance = pd.DataFrame(
        {"feature": X_train_reg.columns, "coefficient": reg.coef_}
    )

    # Sort by absolute value to see the most influential
    feature_importance["abs_coef"] = feature_importance["coefficient"].abs()
    feature_importance = feature_importance.sort_values(by="abs_coef", ascending=False)
    print(feature_importance)

    # Take top 10 by absolute coefficient
    top_features = feature_importance.head(10)

    plt.figure(figsize=(10, 6))
    colors = ["blue" if c > 0 else "red" for c in top_features["coefficient"]]
    plt.barh(top_features["feature"], top_features["coefficient"], color=colors)
    plt.xlabel("Coefficient Value")
    plt.ylabel("Feature")
    plt.title("Top 10 Most Influential Features in Linear Regression", fontsize=14)
    plt.gca().invert_yaxis()  # Largest at top
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    save_fig("Top 10 Most Influential Features in Linear Regression")
    # plt.show() # Uncomment to see figure

    # XGBoost Regression
    xgbreg = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    xgbreg.fit(X_train_reg, y_train_reg)
    y_pred_reg_xgb = xgbreg.predict(X_test_reg)
    mse_xgb = mean_squared_error(y_test_reg, y_pred_reg_xgb)
    rmse_xgb = np.sqrt(mse_xgb)
    r2_xgb = r2_score(y_test_reg, y_pred_reg_xgb)
    print(
        f"XGBoost Regression | MSE: {mse_xgb:.2f}, RMSE: {rmse_xgb:.2f}, r2score: {r2_xgb:.2f}"
    )  # XGBoost Regression | MSE: 22890.67, RMSE: 151.30, r2score: -0.32

    importance_dict = xgbreg.get_booster().get_score(importance_type="weight")
    # Convert to DataFrame
    feature_importance_xgb = pd.DataFrame(
        {
            "feature": list(importance_dict.keys()),
            "importance": list(importance_dict.values()),
        }
    )
    # Sort by importance
    feature_importance_xgb = feature_importance_xgb.sort_values(
        by="importance", ascending=False
    )
    print(feature_importance_xgb.head(10))

    plt.figure(figsize=(10, 8))
    plot_importance(xgbreg, max_num_features=10, importance_type="weight")
    plt.title("Top 10 Feature Importance - XGBoost", fontsize=14)
    save_fig("Top 10 Feature Importance - XGBoost", tight_layout=False)
    # plt.show() # Uncomment to see figure


def classification_analysis(data: pd.DataFrame):
    # The target variable `Customer_Satisfaction` must be left unscaled
    X_clf = data.drop("Customer_Satisfaction", axis=1)
    y_clf = data["Customer_Satisfaction"]

    # Scale numerical features (excluding the target)
    scaler_clf = StandardScaler()
    numeric_features_clf = X_clf.select_dtypes(include=np.number).columns.tolist()
    X_clf[numeric_features_clf] = scaler_clf.fit_transform(X_clf[numeric_features_clf])

    # Split data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    print("Data split into training and testing sets for classification.")

    results = {}

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_clf, y_train_clf)
    y_pred_clf_log = logreg.predict(X_test_clf)
    results["logistic_regression"] = {
        "accuracy": accuracy_score(y_test_clf, y_pred_clf_log),
        "report": classification_report(y_test_clf, y_pred_clf_log),
        "confusion_matrix": confusion_matrix(y_test_clf, y_pred_clf_log),
    }
    print("\nLogistic Regression:")

    # Decision Tree Classifier
    clf_tree = tree.DecisionTreeClassifier(random_state=42)
    clf_tree.fit(X_train_clf, y_train_clf)
    y_pred_clf_tree = clf_tree.predict(X_test_clf)

    results["decision_tree"] = {
        "accuracy": accuracy_score(y_test_clf, y_pred_clf_tree),
        "report": classification_report(y_test_clf, y_pred_clf_tree, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test_clf, y_pred_clf_tree).tolist(),
    }

    print(results)

    return results


if __name__ == "__main__":
    file_path = "https://raw.githubusercontent.com/codeteme/IDS706_DE_Wk2/refs/heads/main/data/raw/Ecommerce_Consumer_Behavior_Analysis_Data.csv"
    data = load_data(file_path)

    cat_cols = [
        "Gender",
        "Income_Level",
        "Marital_Status",
        "Education_Level",
        "Occupation",
        "Purchase_Channel",
        "Social_Media_Influence",
        "Discount_Sensitivity",
        "Engagement_with_Ads",
        "Device_Used_for_Shopping",
        "Payment_Method",
        "Purchase_Intent",
        "Shipping_Preference",
        "Purchase_Category",
    ]

    data = clean_data(data, cat_cols)

    # EDA
    plot_hist(data, "Purchase_Amount", save_name="Purchase_Amount_Hist")
    plot_boxgrid(
        data,
        data.select_dtypes(include=np.number).columns.tolist(),
        save_name="Numerical_Boxplots",
    )
    plot_countgrid(data, cat_cols, save_name="Categorical_Countplots")
    plot_corr_heatmap(
        data,
        data.select_dtypes(include=np.number).columns.tolist(),
        save_name="Correlation_Heatmap",
    )

    data_final = feature_engineering(data)
    regression_analysis(data_final)
    classification_analysis(data_final)
