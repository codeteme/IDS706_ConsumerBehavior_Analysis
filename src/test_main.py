import unittest
import pandas as pd
import numpy as np
from scipy import stats

from main import (
    load_data,
    clean_data,
    exploratory_data_analysis,
    data_transformation_feature_engineering,
    regression_analysis,
    classification_analysis
)


class TestDataPipeline(unittest.TestCase):

    def get_sample_data(self):
        # Minimal sample dataset
        columns = [
            "Customer_ID","Age","Gender","Income_Level","Marital_Status","Education_Level",
            "Occupation","Location","Purchase_Category","Purchase_Amount","Frequency_of_Purchase",
            "Purchase_Channel","Brand_Loyalty","Product_Rating","Time_Spent_on_Product_Research(hours)",
            "Social_Media_Influence","Discount_Sensitivity","Return_Rate","Customer_Satisfaction",
            "Engagement_with_Ads","Device_Used_for_Shopping","Payment_Method","Time_of_Purchase",
            "Discount_Used","Customer_Loyalty_Program_Member","Purchase_Intent","Shipping_Preference",
            "Time_to_Decision"
        ]

        data = [
            ["37-611-6911",22,"Female","Middle","Married","Bachelor's","Middle","Évry","Gardening & Outdoors","$333.80",4,"Mixed",5,5,2,"None","Somewhat Sensitive",1,7,"None","Tablet","Credit Card","3/1/2024",True,False,"Need-based","No Preference",2],
            ["29-392-9296",49,"Male","High","Married","High School","High","Huocheng","Food & Beverages","$222.22",11,"In-Store",3,1,2,"Medium","Not Sensitive",1,5,"High","Tablet","PayPal","4/16/2024",True,False,"Wants-based","Standard",6],
            ["37-611-6911",22,"Female","Middle","Married","Bachelor's","Middle","Évry","Gardening & Outdoors","$333.80",4,"Mixed",5,5,2,"None","Somewhat Sensitive",1,7,"None","Tablet","Credit Card","3/1/2024",True,False,"Need-based","No Preference",2],
            ["29-392-9296",49,"Male","High","Married","High School","High","Huocheng","Food & Beverages","$222.22",11,"In-Store",3,1,2,"Medium","Not Sensitive",1,5,"High","Tablet","PayPal","4/16/2024",True,False,"Wants-based","Standard",6],
            ["37-611-6911",22,"Female","Middle","Married","Bachelor's","Middle","Évry","Gardening & Outdoors","$333.80",4,"Mixed",5,5,2,"None","Somewhat Sensitive",1,7,"None","Tablet","Credit Card","3/1/2024",True,False,"Need-based","No Preference",2],
            ["29-392-9296",49,"Male","High","Married","High School","High","Huocheng","Food & Beverages","$222.22",11,"In-Store",3,1,2,"Medium","Not Sensitive",1,5,"High","Tablet","PayPal","4/16/2024",True,False,"Wants-based","Standard",6],
            ["37-611-6911",22,"Female","Middle","Married","Bachelor's","Middle","Évry","Gardening & Outdoors","$333.80",4,"Mixed",5,5,2,"None","Somewhat Sensitive",1,7,"None","Tablet","Credit Card","3/1/2024",True,False,"Need-based","No Preference",2],
            ["29-392-9296",49,"Male","High","Married","High School","High","Huocheng","Food & Beverages","$222.22",11,"In-Store",3,1,2,"Medium","Not Sensitive",1,5,"High","Tablet","PayPal","4/16/2024",True,False,"Wants-based","Standard",6],
            ["37-611-6911",22,"Female","Middle","Married","Bachelor's","Middle","Évry","Gardening & Outdoors","$333.80",4,"Mixed",5,5,2,"None","Somewhat Sensitive",1,7,"None","Tablet","Credit Card","3/1/2024",True,False,"Need-based","No Preference",2],
            ["29-392-9296",49,"Male","High","Married","High School","High","Huocheng","Food & Beverages","$222.22",11,"In-Store",3,1,2,"Medium","Not Sensitive",1,5,"High","Tablet","PayPal","4/16/2024",True,False,"Wants-based","Standard",6],
            ["37-611-6911",22,"Female","Middle","Married","Bachelor's","Middle","Évry","Gardening & Outdoors","$333.80",4,"Mixed",5,5,2,"None","Somewhat Sensitive",1,7,"None","Tablet","Credit Card","3/1/2024",True,False,"Need-based","No Preference",2],
            ["29-392-9296",49,"Male","High","Married","High School","High","Huocheng","Food & Beverages","$222.22",11,"In-Store",3,1,2,"Medium","Not Sensitive",1,5,"High","Tablet","PayPal","4/16/2024",True,False,"Wants-based","Standard",6],
        ]

        return pd.DataFrame(data, columns=columns)

    def test_load_data_type_and_length(self):
        url = "https://raw.githubusercontent.com/codeteme/IDS706_DE_Wk2/refs/heads/main/data/raw/customers-100.csv"
        df = load_data(url)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_load_data_columns_exist(self):
        df = load_data("https://raw.githubusercontent.com/codeteme/IDS706_DE_Wk2/refs/heads/main/data/raw/customers-100.csv")
        expected_cols = ["Customer_ID", "Age", "Purchase_Amount", "Gender"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_clean_data_no_nulls(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        self.assertEqual(cleaned.isnull().any().sum(), 0)

    def test_clean_data_purchase_amount_float(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        self.assertTrue(cleaned["Purchase_Amount"].dtype == float)

    def test_clean_data_no_outliers(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        numeric_cols = cleaned.select_dtypes(include=np.number)
        z_scores = np.abs(stats.zscore(numeric_cols))
        self.assertEqual(cleaned[(z_scores > 3).any(axis=1)].shape[0], 0)


    def test_exploratory_data_analysis(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        
        self.assertIsInstance(string_cols, list)
        for col in string_cols:
            self.assertTrue(cleaned[col].dtype == "object")


    def test_eda_returns_list_of_strings(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        self.assertIsInstance(string_cols, list)
        for col in string_cols:
            self.assertTrue(cleaned[col].dtype == "object")

    def test_eda_contains_expected_strings(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        expected_strings = ["Gender", "Income_Level", "Marital_Status"]
        for col in expected_strings:
            self.assertIn(col, string_cols)


    def test_feature_engineering_columns_dropped(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        data_final = data_transformation_feature_engineering(cleaned, string_cols)
        for col in ["Customer_ID", "Location", "Time_of_Purchase"]:
            self.assertNotIn(col, data_final.columns)

    def test_feature_engineering_contains_numeric(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        data_final = data_transformation_feature_engineering(cleaned, string_cols)
        self.assertIn("Age", data_final.columns)
        self.assertIn("Purchase_Amount", data_final.columns)

    def test_feature_engineering_one_hot_encoding(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        data_final = data_transformation_feature_engineering(cleaned, string_cols)
        for col in string_cols:
            if col not in ["Customer_ID", "Location", "Time_of_Purchase"]:
                self.assertTrue(any(c.startswith(col + "_") for c in data_final.columns))

    def test_feature_engineering_no_object_columns(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        data_final = data_transformation_feature_engineering(cleaned, string_cols)
        self.assertTrue(all(dtype != "object" for dtype in data_final.dtypes))
    
    def test_regression_structure_and_metrics(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        data_final = data_transformation_feature_engineering(cleaned, string_cols)
        results = regression_analysis(data_final)

        self.assertIn("linear_regression", results)
        self.assertIn("xgboost_regression", results)
        for model, metrics in results.items():
            self.assertGreaterEqual(metrics["mse"], 0)
            self.assertGreaterEqual(metrics["rmse"], 0)
            self.assertAlmostEqual(metrics["rmse"], np.sqrt(metrics["mse"]), places=5)

    def test_regression_prediction_shape(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        data_final = data_transformation_feature_engineering(cleaned, string_cols)
        X_reg = data_final.drop("Purchase_Amount", axis=1)
        results = regression_analysis(data_final)
        self.assertEqual(X_reg.shape[0], len(cleaned))

    def test_classification_structure_and_accuracy(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        data_final = data_transformation_feature_engineering(cleaned, string_cols)
        results = classification_analysis(data_final)

        self.assertIn("logistic_regression", results)
        self.assertIn("decision_tree", results)
        for model, metrics in results.items():
            self.assertGreaterEqual(metrics["accuracy"], 0)
            self.assertLessEqual(metrics["accuracy"], 1)

    def test_classification_confusion_matrix_shape(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        data_final = data_transformation_feature_engineering(cleaned, string_cols)
        results = classification_analysis(data_final)
        for model, metrics in results.items():
            cm = np.array(metrics["confusion_matrix"])
            self.assertEqual(cm.shape[0], cm.shape[1])
            self.assertEqual(cm.shape[0], len(cleaned["Customer_Satisfaction"].unique()))

    def test_classification_target_present_in_split(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        data_final = data_transformation_feature_engineering(cleaned, string_cols)
        y_clf = data_final["Customer_Satisfaction"] if "Customer_Satisfaction" in data_final else cleaned["Customer_Satisfaction"]
        self.assertGreaterEqual(len(y_clf.unique()), 1)


if __name__ == "__main__":
    unittest.main()