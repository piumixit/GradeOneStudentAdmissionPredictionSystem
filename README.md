# Grade One Student Admission Prediction System

An AI application that predicts next year's Grade 1 student admissions in Sri Lanka using historical admission data.

**1. Problem Statement**

Planning Grade 1 admissions is important for allocating classrooms, teachers, learning materials, and other school resources.

This project uses historical admission data to predict future Grade 1 admissions by province, district, medium, and gender.

**2. Use Case**

The system can be used by:

Education authorities
School administrators
Education planners
Researchers

Users can select a province, district, medium, and gender to get the predicted number of Grade 1 admissions for the next year.

**3. Solution Overview**

The application uses historical admission data from 2022–2024 and a machine learning regression model to predict future admissions.


**4. Dataset**

Dataset: Grade One New Admissions by Gender & Medium of Study

Source: Gov. Data Management Portal Educational Statistics

The dataset contains:

Year
Province
District
Medium
Gender
Admissions

File:

admission.csv


Years used:

2022, 2023, 2024

**5. AI/ML Approach**

This is a regression problem where the target variable is Admissions.

Model

Random Forest Regressor

Features
Year
Province
District
Medium
Gender
Libraries
Python
Pandas
NumPy
Scikit-learn
Joblib

**6. Application Architecture**
 Admission Data
      ->
Data Processing
      ->
Machine Learning Model
      ->
Admission Prediction
      ->
Streamlit Web App


**7. Technology Stack**

Python -	Application & ML,
Pandas	- Data processing,
Scikit-learn -	ML model,
Streamlit -	Web application,
GitHub	- Source code,
Streamlit - Cloud	Deployment

**8. Local Setup**

Clone the repository:

git clone (https://github.com/piumixit/GradeOneStudentAdmissionPredictionSystem/tree/main)
cd gradeonestudentadmissionpredictionsystem

Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py

The application runs on:
http://localhost:8501

**9. Deployment**

The web application is deployed using Streamlit Cloud.

Deployment steps:

Push the project to GitHub.
Connect the repository to Streamlit Cloud.
Select app.py.
Deploy the application.

**10. Web Application Usage**
Select a Province.
Select a District.
Select the Medium.
Select the Gender.
Select the prediction year.
Click Predict.
View the estimated Grade 1 admissions.

Live application: https://gradeonestudentadmissionpredictionsystem.streamlit.app/
