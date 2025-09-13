# IDS706_DE_Wk2 / Week 3 Submission

## E-commerce Consumer Behavior Analysis

### Project Overview
This project is a comprehensive analysis of e-commerce consumer behavior data. It involves several key steps in the data science pipeline: loading and cleaning data, performing exploratory data analysis (EDA), and building and evaluating machine learning models for regression and classification tasks.

**Week 3 Goal:** Make the project **reproducible and testable** using Docker and/or Dev Container. Implement **unit tests** for core data analysis functions and ensure environment setup is consistent across systems.  

---

## Dataset
The analysis uses the dataset:

```

data/raw/Ecommerce\_Consumer\_Behavior\_Analysis\_Data.csv

````

It contains demographics, purchase history, and behavioral metrics for e-commerce customers.  

---

## Previous Project Results
- **Data Cleaning:** handled missing values, outliers, duplicates, and type conversions.  
- **Data Transformation:** standardization and one-hot encoding for ML models.  
- **EDA:** histograms, count plots, correlation heatmaps, and group analysis.  
- **Machine Learning Models:**  
  - Regression (`LinearRegression`, `XGBRegressor`) predicting `Purchase_Amount`.  
  - Classification (`LogisticRegression`, `DecisionTreeClassifier`) predicting `Customer_Satisfaction`.  
- **Findings:** Models struggled to accurately predict targets, suggesting complex relationships or additional features are needed.  

---

## Setup Instructions

### Docker Image
Build the Docker image with all dependencies:

```bash
docker build -t ids706_wk3 .
````

* Uses Python 3.10 and `requirements.txt`.
* Isolated environment ensures reproducibility.

---

### Run Tests

Run container to execute unit tests:

```bash
docker run --rm ids706_wk3
```

* Tests are located in `src/test_main.py`.
* Command executed:

```bash
python -m unittest discover -s src
```

* Screenshot of passing tests should be included for submission.

---

## Dev Container Setup (VS Code)

1. **Folder structure**:

```
.devcontainer/
├── devcontainer.json
Dockerfile
requirements.txt
src/
data/
```

2. **devcontainer.json**:

```json
{
  "name": "IDS706 Week 3 Dev Container",
  "build": {
    "dockerfile": "../Dockerfile",
    "context": ".."
  },
  "workspaceFolder": "/app",
  "extensions": [
    "ms-python.python",
    "ms-python.vscode-pylance"
  ],
  "settings": {
    "python.pythonPath": "/usr/local/bin/python"
  },
  "postCreateCommand": "pip install --no-cache-dir -r requirements.txt"
}
```

3. **Open in VS Code:**

* Install **Remote - Containers** extension.
* Press `F1` → **Remote-Containers: Open Folder in Container** → select project folder.
* Dependencies install automatically and workspace opens inside the container.

4. **Run Tests inside Dev Container:**

```bash
python -m unittest discover -s src
```

---

## Figures

![Alt text for screen readers](assets/img001.png)
![Alt text for screen readers](assets/img002.png)
![Alt text for screen readers](assets/img003.png)
![Alt text for screen readers](assets/img004.png)
![Alt text for screen readers](assets/img005.png)