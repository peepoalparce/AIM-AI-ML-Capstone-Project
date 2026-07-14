#Credit Card Fraud Detection — AIM AI/ML Capstone Project
This project implements a full machine‑learning pipeline for detecting fraudulent credit card transactions. It follows industry best practices for data preprocessing, model training, evaluation, explainability, and fairness analysis.

##Project Overview
 Credit card fraud is rare but financially damaging. This project builds a machine‑learning system capable of identifying fraudulent transactions using:
 -PCA‑transformed anonymized features (V1–V28)
 -Transaction time
 -Transaction amount
 The dataset is highly imbalanced, so techniques like SMOTE, class weighting, and threshold tuning are applied to improve fraud detection performance.

##Features Implemented
1. Data Preprocessing
- Missing value checks
- Duplicate detection
- Outlier analysis (IQR method)
- Standard scaling for Time and Amount
- SMOTE oversampling
2. Feature Selection
- Random Forest feature importance
- SelectFromModel (median threshold)
3. Dimensionality Reduction & Visualization
- PCA (2D projection)
- t‑SNE (2D projection)
4. Models Trained
- Logistic Regression
- Random Forest
- XGBoost
5. Evaluation Metrics
- Classification report
- Confusion matrix
- ROC‑AUC
- Precision‑Recall AUC
- Threshold tuning
6. Explainability
- Permutation importance
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE)
- LIME explanations for fraud cases
7. Fairness Analysis
- High vs low transaction amount groups
- TPR, FPR, positive rate
- Demographic parity ratio
- Equalized odds
- Disparate impact

##Results Summary
The Random Forest and XGBoost models typically achieve:
- High ROC‑AUC
- Strong PR‑AUC
- Improved recall for fraud cases
- Balanced performance across transaction amount groups
Exact results are included in the /reports/ folder.

##How to Run the Project
1. Install dependencies
pip install -r requirements.txt
2. Run the full pipeline
python src/main.py
3. View generated outputs
Saved models → /models/
Reports → /reports/
Visualizations → displayed during execution 

##Dataset
The dataset used is the Credit Card Fraud Detection Dataset from Kaggle.
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

##Author
Orlando Pedro Z. Alparce  
AIM Postgraduate Diploma in Artificial Intelligence & Machine Learning






