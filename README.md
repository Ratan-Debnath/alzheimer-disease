# Alzheimer Disease Prediction using Gene Expression Data
This project aims to predict Alzheimer’s Disease (AD) using a logistic regression model and other machine learning classifiers on gene expression data. The dataset comprises gene expression levels from subjects labeled as either AD (Alzheimer's Disease) or CTL (Control). The models help classify subjects based on gene activity and identify potential biomarkers for early detection.

# Dataset
Source: Custom CSV file stored in Google Drive

Samples: Each row represents a subject

Features: 430+ gene expression levels

Target: Diagnosis (AD or CTL)

# Machine Learning Models Used
Model	Accuracy
Logistic Regression	87.88%
SVM (Linear Kernel)	84.85%
SVM (RBF Kernel)	72.73%
K-Nearest Neighbors	75.76%
Decision Tree	63.64%
Random Forest	84.85%

# Project Workflow
Data Loading

Mounted Google Drive

Loaded the gene expression dataset

Data Preprocessing

Removed ID column

Checked for null values

Split data into training and test sets

Model Training & Evaluation

Applied multiple classification models

Measured performance using accuracy, confusion matrix, precision, recall, and F1-score

# Evaluation Metrics
Each model was evaluated using:

Accuracy

Precision / Recall / F1-Score

Confusion Matrix

Classification Report

# Key Findings
Logistic Regression achieved the highest accuracy and best balance between precision and recall.

Random Forest also performed competitively and could capture non-linear patterns.

SVM with RBF Kernel underperformed likely due to data dimensionality or kernel parameters.

# Tech Stack
Python (Pandas, NumPy, Scikit-learn, Matplotlib)

Jupyter Notebook (Colab)

Google Drive (for dataset storage)

# Future Improvements
Implement feature selection or dimensionality reduction (e.g., PCA)

Use ensemble techniques like XGBoost or stacking classifiers

Perform hyperparameter tuning (e.g., GridSearchCV)

Test on larger or external datasets for validation
