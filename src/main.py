# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from sklearn.linear_model import LogisticRegression
from sklearn import tree

from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

# %%
data = pd.read_csv("../data/raw/Ecommerce_Consumer_Behavior_Analysis_Data.csv")
data.head()

# %%
data.info()

# %%
data.describe()

# %% [markdown]
# ### Data Cleaning

# %%
z_scores = np.abs(stats.zscore(data.select_dtypes(include=np.number)))
data[(z_scores > 3).all(axis = 1)] # There are no outliers 

# %%
for col in data.columns:
    if data[col].isnull().sum() > 0:
        print(f"{col} has {data[col].isnull().sum()} null values")

# %%
data['Social_Media_Influence'] = data['Social_Media_Influence'].fillna(data['Social_Media_Influence'].mode()[0]) # Replace null values with model
data['Engagement_with_Ads'] = data['Engagement_with_Ads'].fillna(data['Engagement_with_Ads'].mode()[0])  # Replace null values with model

print(f"Total number of columns with missing values after clean up: {data.isnull().any().sum()}")

data.duplicated().value_counts() # There are no duplicated columns

# %%
# Looking at data.info() and checking against data, 'Purchase_Amount' could be converted to a float
data['Purchase_Amount'] = data['Purchase_Amount'].str.replace('$', '').astype(float)
assert data['Purchase_Amount'].dtype == 'float64'

# %%
# Data type conversions
numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
string_cols = data.select_dtypes(include="object").columns

# %%
data.groupby('Income_Level')['Purchase_Amount'].mean()

# %%
plt.figure(figsize=(10, 5))
sns.histplot(x = data['Age'])
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# %%
plt.figure(figsize=(15, 5))
sns.countplot(x = data['Gender'])
plt.xlabel("Frequency")
plt.ylabel("Age")
plt.show()

# %%
plt.figure(figsize=(8, 5))
sns.countplot(x = data['Marital_Status'])
plt.xlabel('Martial Status')
plt.ylabel('Frequency')
plt.show()

# %%
plt.figure(figsize=(12, 8))
sns.heatmap(data[numeric_cols].corr())
plt.show()


# %% [markdown]
# ### Data Transformation

# %%
# No need to apply scaler to the dependent variables we are going to work on
scaler = StandardScaler()
cols_to_scale = data[numeric_cols].drop(['Purchase_Amount', 'Customer_Satisfaction'], axis = 1).columns 
data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

# %%
# One-hot encode 
cols_to_onehot = data[string_cols].drop(['Customer_ID', 'Time_of_Purchase', 'Location'], axis = 1).columns.tolist()
data = pd.get_dummies(data, columns=cols_to_onehot, drop_first=True)

# %% [markdown]
# ### Explore a Machine Learning Algorithm
# 

# %% [markdown]
# #### LinearRegression

# %%
X = data.drop(['Customer_ID', 'Location', 'Time_of_Purchase', 'Purchase_Amount'], axis = 1) # These don't have predictive power or are DV
y = data['Purchase_Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = LinearRegression().fit(X_train, y_train)

# Test against the test 
y_prediction = reg.predict(X_test)

mse = mean_squared_error(y_test, y_prediction)
rmse = np.sqrt(mse)
print(f"MSE = {mse}. RMSE = {rmse}.") # MSE = 18819.681327525523. RMSE = 137.18484365091328.
# The resulting MSE is quite high. The model might as well just choose the mean value for every given input.

# %%
importance = reg.coef_ 
features = reg.feature_names_in_

plt.figure(figsize=(8, 5))
plt.barh(features, importance)
plt.xlabel("Importance Score")
plt.ylabel("Feature Importance")
plt.show()

# %% [markdown]
# #### XGBRegressor

# %%
xgbreg = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgbreg.fit(X_train, y_train)

y_prediction = xgbreg.predict(X_test)

mse = mean_squared_error(y_test, y_prediction)
rmse = np.sqrt(mse)
print(f"MSE = {mse}. RMSE = {rmse}.") # MSE = 22890.666174973165. RMSE = 151.29661653511346.

# %%
xgb_importance = xgbreg.feature_importances_ 
xgb_features = X.columns

plt.figure(figsize=(15, 10))
plt.barh(xgb_features, xgb_importance)
plt.xlabel("Importance Score")
plt.ylabel("Feature Importance")
plt.show()

# %% [markdown]
# ### Classification

# %%
# Scale Purchase Amount. Now it will be used as a feature. 
data['Purchase_Amount'] = scaler.fit_transform(data[['Purchase_Amount']])

# Change the column 'Customer_Satisfaction' into categorical column 
data['Customer_Satisfaction'] = data['Customer_Satisfaction'].astype('category')

# %% [markdown]
# #### Logistic Regression

# %%
logreg = LogisticRegression()

X = data.drop(['Customer_ID', 'Location', 'Time_of_Purchase', 'Customer_Satisfaction'], axis = 1)
y = data['Customer_Satisfaction']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg.fit(X_train, y_train)
y_prediction = logreg.predict(X_test)

mse = mean_squared_error(y_test, y_prediction)
rmse = np.sqrt(mse)
print(f"MSE = {mse}. RMSE = {rmse}.")

acc_score = accuracy_score(y_test, y_prediction)
print(f"Prediction accuracy: {acc_score}")

# MSE = 15.03. RMSE = 3.8768543949960255.
# Prediction accuracy: 0.07

# %%
clf = tree.DecisionTreeClassifier()

clf.fit(X_train, y_train)
y_prediction = clf.predict(X_test)

mse = mean_squared_error(y_test, y_prediction)
rmse = np.sqrt(mse)
print(f"MSE = {mse}. RMSE = {rmse}.")

acc_score = accuracy_score(y_test, y_prediction)
print(f"Prediction accuracy: {acc_score}")

# MSE = 14.995. RMSE = 3.8723377951826463.
# Prediction accuracy: 0.145


