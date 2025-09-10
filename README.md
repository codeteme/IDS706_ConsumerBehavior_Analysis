# IDS706_DE_Wk2

# E-commerce Consumer Behavior Analysis
## Project Overview
This project is a comprehensive analysis of e-commerce consumer behavior data. It involves several key steps in the data science pipeline: loading and cleaning data, performing exploratory data analysis (EDA), and building and evaluating several machine learning models for both regression and classification tasks.

The primary goals of this project are to understand the factors influencing consumer purchases and satisfaction, and to experiment with different machine learning algorithms to predict these behaviors.

## Dataset
The analysis is based on the "Ecommerce_Consumer_Behavior_Analysis_Data.csv" dataset. This dataset contains information on e-commerce customers, including demographics, purchase history, and various behavioral metrics.

## Getting Started
To run this analysis, you will need to have Python and the following libraries installed. You can install them using pip:

```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

You will also need to ensure the dataset is placed in the correct directory as specified in the code: 
`../data/raw/Ecommerce_Consumer_Behavior_Analysis_Data.csv.`

## Analysis Steps
### 1. Data Loading and Inspection
The analysis begins by importing all necessary libraries, followed by loading the dataset into a pandas DataFrame. The `.head()`, `.info()`, and `.describe()` methods are used to get a quick overview of the data's structure, types, and summary statistics.

### 2. Data Cleaning and Preprocessing
This crucial step prepares the data for modeling:

* Outlier Detection: The code checks for outliers in numeric columns using the Z-score method.

* Missing Values: Null values in the `Social_Media_Influence` and `Engagement_with_Ads` columns are identified and filled using the mode of each column.

* Duplicates: The dataset is checked for any duplicate rows, of which none were found.

* Data Type Conversion: The `Purchase_Amount` column is converted from a string to a float by removing the '$' symbol.

* Data Transformation:

* * Standardization: Key numeric features are scaled using `StandardScaler` to ensure they contribute equally to the models.

* * One-Hot Encoding: Categorical string columns are converted into a numerical format using one-hot encoding, a necessary step for most machine learning algorithms.

### 3. Exploratory Data Analysis (EDA)
Several visualizations and groupings were performed to understand the data better:

* A histogram of `Age` and count plots for `Gender` and `Marital_Status` were created to understand the distribution of these demographic features.

* The average `Purchase_Amount` for different `Income_Level` groups was calculated.

* A correlation heatmap was generated to visualize the relationships between numeric variables.

### 4. Machine Learning Models
Both regression and classification models were explored.

**Regression Analysis (Predicting Purchase_Amount)**
* Linear Regression: A `LinearRegression` model was trained to predict the `Purchase_Amount`. The resulting Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) were quite high, suggesting that a simple linear model is not a good fit for this data. Feature importance was also plotted.
* XGBoost Regressor: An `XGBRegressor` was used as a more complex alternative. The results showed a slightly higher error, indicating that even this advanced model struggled to capture the underlying patterns in the current feature set.

**Classification Analysis (Predicting Customer_Satisfaction)**

* Logistic Regression: The `Purchase_Amount` column was scaled, and `Customer_Satisfaction` was converted to a categorical target. A `LogisticRegression` model was trained to predict customer satisfaction. The model achieved a very low prediction accuracy score of about 7%.
* Decision Tree Classifier: A `DecisionTreeClassifier` was also tested, which resulted in a slightly better but still low accuracy of about 14.5%.

### Conclusion
The analysis reveals that while the data was successfully cleaned and prepared, the selected features and initial models struggled to accurately predict both `Purchase_Amount` and `Customer_Satisfaction`. The low accuracy scores suggest that either the relationship between the features and the target variables is complex, or that additional data features are needed to improve model performance.

=====

#### Figures from python script
![Alt text for screen readers](assets/img001.png)
========
========
![Alt text for screen readers](assets/img002.png)
========
========
![Alt text for screen readers](assets/img003.png)
========
========
![Alt text for screen readers](assets/img004.png)
========
========
![Alt text for screen readers](assets/img005.png)