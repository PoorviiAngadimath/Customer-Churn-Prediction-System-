# Customer Churn Prediction System

## Project Overview

The Customer Churn Prediction System is a machine learning project developed during my Data Science Internship at TakeItSmart. The project predicts whether a customer is likely to leave a service based on customer-related information.

## Project Objectives

- Predict customer churn using machine learning.
- Analyse customer information and behaviour.
- Identify customers who are likely to discontinue a service.
- Help businesses understand customer churn patterns.

## Technologies Used

- **Programming Language:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Class Imbalance Handling:** SMOTE (Imbalanced-learn)
- **Web Application:** Streamlit

## Project Workflow

1. Loaded the customer churn dataset using Pandas.
2. Performed data preprocessing.
3. Handled missing values and duplicate records.
4. Encoded categorical variables.
5. Scaled numerical features.
6. Applied SMOTE to handle class imbalance.
7. Trained and evaluated multiple machine learning models.
8. Selected Random Forest as the final model.
9. Saved the trained model and supporting preprocessing objects.
10. Developed a Streamlit application for customer churn prediction.

## Machine Learning Models

Explored and evaluated the following algorithms:

- Random Forest
- AdaBoost
- Extra Trees
- Gradient Boosting
- Voting Classifier
- Stacking Classifier

**Final Model:** Random Forest

## Project Files

 File - Description 
 app.py - Streamlit application for churn prediction 
 model.pkl - Saved trained machine learning model 
 scaler.pkl - Saved feature scaler 
 features.pkl - Saved feature information 
 telecom_churn_data .csv - Customer churn dataset 
 README.md - Project documentation 

## How to Run the Project

### Step 1: Clone the repository
git clone https://github.com/PoorviiAngadimath/Customer-Churn-Prediction-System-.git

### Step 2: Navigate to the project folder
cd Customer-Churn-Prediction-System-

### Step 3: Install the required libraries
pip install streamlit pandas numpy scikit-learn imbalanced-learn

### Step 4: Run the application
streamlit run app.py

## Applications
- Customer churn prediction
- Customer behaviour analysis
- Identifying potential customer loss
- Supporting customer retention decisions

## Author
**Poorvi Angadimath**
BE – Computer Science and Engineering (Artificial Intelligence & Machine Learning)

**GitHub:** [PoorviiAngadimath](https://github.com/PoorviiAngadimath)

Developed as part of my Data Science Internship at TakeItSmart (February–May 2026).
