# 🏥 Diabetes Prediction Model - Logistic Regression Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Library](https://img.shields.io/badge/Imbalanced--Learn-SMOTE-green)

## 📋 Project Overview
This project implements a Machine Learning pipeline to predict the likelihood of diabetes in patients based on diagnostic measures. 

Unlike standard accuracy-focused models, this analysis prioritizes **clinical relevance**. It specifically addresses class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)** and optimizes the decision threshold to maximize **Recall**, ensuring that potential diabetes cases are not missed (minimizing False Negatives).

## ⚙️ Key Features
The analysis pipeline includes the following stages:

1.  **Data Preprocessing**: 
    * Detection and handling of invalid zero values in biological indicators (Glucose, BMI, Insulin, etc.) using median imputation.
2.  **Feature Engineering**: 
    * Creation of derived features such as `Age_Group`, `BMI_Category`, `Glucose_Risk`, and `Insulin_Resistance` indices.
3.  **Class Imbalance Handling**: 
    * Implementation of **SMOTE** to generate synthetic samples for the minority class (Diabetic cases).
4.  **Model Optimization**: 
    * **Logistic Regression** with class weights and regularization (`L2`).
    * **Threshold Tuning**: moving the decision boundary to maximize the F1-Score and Recall.
5.  **Comprehensive Evaluation**:
    * ROC-AUC Score, Confusion Matrix, and Precision-Recall Curves.
    * Feature Importance analysis based on regression coefficients.
6.  **Clinical Simulation**:
    * A practical module that tests the model against hypothetical patient profiles (Low, Medium, High Risk).

## 🛠️ Technologies Used
* **Python 3.x**
* **Pandas & NumPy**: Data manipulation and linear algebra.
* **Scikit-Learn**: Model building, scaling, and evaluation.
* **Imbalanced-Learn**: SMOTE implementation.
* **Matplotlib & Seaborn**: Data visualization.

## 📂 Dataset
The model expects a CSV file named `diabetes.csv` containing the following columns:
* `Pregnancies`
* `Glucose`
* `BloodPressure`
* `SkinThickness`
* `Insulin`
* `BMI`
* `DiabetesPedigreeFunction`
* `Age`
* `Outcome` (Target: 0 or 1)

*(Note: This is typically based on the Pima Indians Diabetes Database)*

## 🚀 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/diabetes-prediction-model.git](https://github.com/yourusername/diabetes-prediction-model.git)
cd diabetes-prediction-model
