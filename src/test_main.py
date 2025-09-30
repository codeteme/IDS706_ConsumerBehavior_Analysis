import unittest
import pandas as pd
import numpy as np
from main import load_data, clean_data, feature_engineering

class TestEcommercePipeline(unittest.TestCase):

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
            ["29-392-9296",49,"Male","High","Married","High School","High","Huocheng","Food & Beverages","$222.22",11,"In-Store",3,1,2,"Medium","Not Sensitive",1,5,"Tablet","PayPal","4/16/2024",True,False,"Wants-based","Standard",6],
            ["45-123-4567",35,"Female","Low","Single","High School","Low","Paris","Clothing","$150.00",7,"Online",4,4,3,"High","Very Sensitive",2,8,"Medium","Mobile","Credit Card","5/10/2024",False,True,"Need-based","Standard",3],
            ["56-987-6543",28,"Male","Middle","Single","Bachelor's","Middle","Lyon","Electronics","$500.50",2,"Online",2,3,1,"Medium","Somewhat Sensitive",0,9,"High","Laptop","PayPal","6/5/2024",True,False,"Wants-based","No Preference",5],
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
        cat_cols = ["Gender", "Income_Level", "Marital_Status", "Education_Level",
                    "Occupation", "Purchase_Channel", "Social_Media_Influence",
                    "Discount_Sensitivity", "Engagement_with_Ads", "Device_Used_for_Shopping",
                    "Payment_Method", "Purchase_Intent", "Shipping_Preference",
                    "Purchase_Category"]
        cleaned = clean_data(df, cat_cols)
        # Only check that the columns we impute are not null
        self.assertEqual(cleaned["Social_Media_Influence"].isnull().sum(), 0)
        self.assertEqual(cleaned["Engagement_with_Ads"].isnull().sum(), 0)


    def test_feature_engineering_columns_dropped(self):
        df = self.get_sample_data()
        cat_cols = ["Gender", "Income_Level", "Marital_Status", "Education_Level",
                    "Occupation", "Purchase_Channel", "Social_Media_Influence",
                    "Discount_Sensitivity", "Engagement_with_Ads", "Device_Used_for_Shopping",
                    "Payment_Method", "Purchase_Intent", "Shipping_Preference",
                    "Purchase_Category"]
        cleaned = clean_data(df, cat_cols)
        data_final = feature_engineering(cleaned)
        # Check that object/datetime columns are dropped
        for col in ["Customer_ID", "Location", "Time_of_Purchase"]:
            self.assertNotIn(col, data_final.columns)
        # Ensure numeric columns remain
        for col in ["Age", "Purchase_Amount"]:
            self.assertIn(col, data_final.columns)

if __name__ == "__main__":
    unittest.main()
