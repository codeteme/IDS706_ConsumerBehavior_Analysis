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

    def test_load_data(self):
        url = "https://raw.githubusercontent.com/codeteme/IDS706_DE_Wk2/refs/heads/main/data/raw/customers-100.csv"
        df = load_data(url)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_clean_data(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        
        self.assertEqual(cleaned.isnull().any().sum(), 0)
        numeric_cols = cleaned.select_dtypes(include=np.number)
        z_scores = np.abs(stats.zscore(numeric_cols))
        # No rows with all numeric z-scores > 3
        self.assertEqual(cleaned[(z_scores > 3).all(axis=1)].shape[0], 0)

    def test_exploratory_data_analysis(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        
        self.assertIsInstance(string_cols, list)
        for col in string_cols:
            self.assertTrue(cleaned[col].dtype == "object")

    def test_data_transformation_feature_engineering(self):
        df = self.get_sample_data()
        cleaned = clean_data(df)
        string_cols = exploratory_data_analysis(cleaned)
        
        data_final = data_transformation_feature_engineering(cleaned, string_cols)

        for col in ["Customer_ID", "Location", "Time_of_Purchase"]:
            self.assertNotIn(col, data_final.columns)

        self.assertIn("Age", data_final.columns)
        self.assertIn("Purchase_Amount", data_final.columns)

        # Check that one-hot encoding occurred
        for col in string_cols:
            if col not in ["Customer_ID", "Location", "Time_of_Purchase"]:
                self.assertTrue(any(c.startswith(col + "_") for c in data_final.columns))
    
    def test_regression_analysis_runs(self):
        # Regression should run without error
        try:
            df = self.get_sample_data()
            cleaned = clean_data(df)
            string_cols = exploratory_data_analysis(cleaned)
            data_final = data_transformation_feature_engineering(cleaned, string_cols)

            regression_analysis(data_final)
        except Exception as e:
            self.fail(f"regression_analysis raised an exception {e}")

    def test_classification_analysis_runs(self):
        # Classification should run without error
        try:
            df = self.get_sample_data()
            cleaned = clean_data(df)
            string_cols = exploratory_data_analysis(cleaned)
            data_final = data_transformation_feature_engineering(cleaned, string_cols)
            
            classification_analysis(data_final)
        except Exception as e:
            self.fail(f"classification_analysis raised an exception {e}")

if __name__ == "__main__":
    unittest.main()