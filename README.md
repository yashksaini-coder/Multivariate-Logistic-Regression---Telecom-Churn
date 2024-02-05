# Telecom Churn Prediction with Multivariate Logistic Regression

## Overview

In this project, we aim to build a predictive model for telecom churn prediction using multivariate logistic regression. The dataset contains information on various customer attributes, including demographics, services availed, and expenses. The primary goal is to predict whether a customer will churn or not, where 'Churn' is a binary variable: 1 denotes that the customer has churned, and 0 denotes that the customer has not churned.

## Dataset Description
| S.No. | Variable Name       | Meaning                                                           |
|-------|---------------------|-------------------------------------------------------------------|
| 1     | CustomerID          | The unique ID of each customer                                    |
| 2     | Gender              | The gender of a person                                           |
| 3     | SeniorCitizen       | Whether a customer can be classified as a senior citizen          |
| 4     | Partner             | If a customer is married/ in a live-in relationship               |
| 5     | Dependents          | If a customer has dependents (children/ retired parents)          |
| 6     | Tenure              | The time for which a customer has been using the service          |
| 7     | PhoneService        | Whether a customer has a landline phone service along with the internet service |
| 8     | MultipleLines       | Whether a customer has multiple lines of internet connectivity    |
| 9     | InternetService     | The type of internet services chosen by the customer               |
| 10    | OnlineSecurity      | Specifies if a customer has online security                        |
| 11    | OnlineBackup        | Specifies if a customer has online backup                          |
| 12    | DeviceProtection    | Specifies if a customer has opted for device protection           |
| 13    | TechSupport         | Whether a customer has opted for tech support or not               |
| 14    | StreamingTV         | Whether a customer has an option of TV streaming                   |
| 15    | StreamingMovies     | Whether a customer has an option of Movie streaming                |
| 16    | Contract            | The type of contract a customer has chosen                         |
| 17    | PaperlessBilling    | Whether a customer has opted for paperless billing                 |
| 18    | PaymentMethod       | Specifies the method by which bills are paid                       |
| 19    | MonthlyCharges      | Specifies the money paid by a customer each month                  |
| 20    | TotalCharges        | The total money paid by the customer to the company                |
| 21    | Churn               | This is the target variable which specifies if a customer has churned or not |

---
## Target Variable

The target variable is 'Churn,' indicating whether a particular customer has churned or not. It is a binary variable:
- 1:- Customer has churned
- 0:- Customer has not churned

# Objective

The objective is to develop a robust Multivariate logistic regression model that can accurately predict customer churn based on historical data. By analyzing past information, the model will be trained to identify patterns and relationships between customer attributes and the likelihood of churn.

# Steps:

1. **Data Exploration:**
   - Explore and understand the dataset.
   - Check for missing values, outliers, and data distribution.

2. **Data Preprocessing:**
   - Handle missing values and outliers.
   - Encode categorical variables.
   - Scale or normalize numerical features.

3. **Model Building:**
   - Split the dataset into training and testing sets.
   - Build a multivariate logistic regression model using the training data.

4. **Model Evaluation:**
   - Evaluate the model's performance on the testing set.
   - Analyze key metrics such as accuracy, precision, recall, and F1 score.

5. **Model Interpretation:**
   - Interpret the coefficients of the logistic regression model to understand the impact of each feature on the likelihood of churn.

---
# Methods and Techniques:-

1. **Data Import and Merging:-**
   - Imported necessary libraries, including Pandas and NumPy.
   - Loaded multiple datasets related to telecom customer information.
   - Merged the datasets using the 'customerID' as a common key.

2. **Data Inspection and Preparation:-**
   - Explored the head, dimensions, and statistical aspects of the merged dataset.
   - Converted binary variables ('Yes/No') to numeric (0/1).
   - Created dummy variables for categorical features using one-hot encoding.
   - Handled missing values by removing observations with missing 'TotalCharges'.
   - Checked for outliers in continuous variables ('tenure', 'MonthlyCharges', 'SeniorCitizen', 'TotalCharges').

3. **Train-Test Split:-**
   - Split the dataset into training and testing sets using the `train_test_split` method.

4. **Feature Scaling:-**
   - Used StandardScaler to scale numerical features ('tenure', 'MonthlyCharges', 'TotalCharges').

5. **Correlation Analysis:-**
   - Explored the correlation between different features.
   - Dropped highly correlated dummy variables to avoid multicollinearity.

6. **Logistic Regression Model Building:-**
   - Used StatsModels' Generalized Linear Model (GLM) for logistic regression.
   - Iteratively performed feature selection using Recursive Feature Elimination (RFE).
   - Checked for Variance Inflation Factors (VIF) to identify and remove multicollinearity.
   - Checked confusion matrices, accuracy, sensitivity, specificity, precision, and recall.

7. **ROC Curve and AUC:-**
   - Plotted the Receiver Operating Characteristic (ROC) curve.
   - Calculated the Area Under the Curve (AUC) for model evaluation.

8. **Optimal Cutoff Point:-**
   - Determined the optimal cutoff probability using accuracy, sensitivity, and specificity.
   - Adjusted the predicted probabilities based on the chosen cutoff.

9. **Precision and Recall:-**
   - Calculated precision and recall for model evaluation.
   - Explored the precision-recall tradeoff.

10. **Making Predictions on Test Set:-**
   - Applied the trained model on the test set.
   - Explored different probability cutoffs and assessed accuracy, sensitivity, and specificity.

---
# Explanations:-

- **Logistic Regression:-**
  - Used for binary classification problems (Churn/Not Churn).

- **One-Hot Encoding:-**
  - Technique to convert categorical variables into binary (dummy) variables.

- **Recursive Feature Elimination (RFE):-**
  - Method for selecting features by recursively removing the least important ones.

- **Receiver Operating Characteristic (ROC) Curve:-**
  - Graphical representation of the tradeoff between sensitivity and specificity.

- **Area Under the Curve (AUC):-**
  - Measures the area under the ROC curve, indicating model performance.

- **Variance Inflation Factor (VIF):-**
  - Checks for multicollinearity in the regression model.

- **Confusion Matrix:-**
  - Table used for evaluating the performance of a classification model.

- **Precision and Recall:-**
  - Metrics for evaluating the predictive power of a model in binary classification.

- **Feature Scaling (StandardScaler):-**
  - Normalizes numerical features for better convergence in logistic regression.

- **Train-Test Split:-**
  - Divides the dataset into training and testing subsets for model evaluation.

- **Dummy Variables:-**
  - Binary columns created to represent categorical data.

---
# Usage

### Clone the Repository:

1. Open your terminal or command prompt.

2. Navigate to the directory where you want to store the project:
   ```bash
   cd path/to/your/directory
   ```

3. Clone the repository:
   ```bash
   git clone https://github.com/yashksaini-coder/Multivariate-Logistic-Regression---Telecom-Churn
   ```

### Navigate to the Project:

```bash
cd telecom-churn-prediction
```

### Set Up the Virtual Environment (Optional but Recommended):

```bash
python -m venv venv
```

### Activate the Virtual Environment:

- **For Windows:**
  ```bash
  .\venv\Scripts\activate
  ```
- **For macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

### Install Dependencies:

```bash
pip install -r requirements.txt
```

### Run the Jupyter Notebook:

```bash
jupyter-notebook
```

This will open a new tab in your web browser showing the Jupyter Notebook interface. Navigate to the cloned project directory and open the notebook titled `Logistic-Regression.ipynb`.

### Execute the Code:-

Run each cell in the notebook sequentially. Ensure that you have the necessary datasets (`churn_data.csv`, `customer_data.csv`, `internet_data.csv`, and `Dictionary.csv`) in the `Datasets` folder.
