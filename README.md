# CODSOFT
# Data Science Projects

This repository contains three data science projects:

1. **Titanic Survival Prediction**
2. **Iris Flower Classification**
3. **Credit Card Fraud Detection**

Each project applies different data science techniques and datasets to solve real-world problems.

---
## 1. Titanic Survival Prediction
### Overview
The Titanic dataset contains information about passengers, including their age, class, gender, and survival status. This project aims to predict whether a passenger survived or not based on given features.

### Dataset
- **Source**: [Kaggle - Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- **Features**:
  - Passenger Class
  - Name, Age, Gender
  - Siblings/Spouses Aboard, Parents/Children Aboard
  - Ticket Fare
  - Survival Status (Target variable)

### Methodology
- Data Preprocessing (Handling missing values, encoding categorical data, feature scaling)
- Exploratory Data Analysis (EDA) using Matplotlib and Seaborn
- Model Training: Logistic Regression, Decision Trees, Random Forest, SVM
- Performance Evaluation (Accuracy, Precision, Recall, F1-score)

### Tools & Libraries
- Python, Pandas, NumPy
- Scikit-Learn, Matplotlib, Seaborn

---
## 2. Iris Flower Classification
### Overview
The Iris dataset is a classic dataset in data science used for classification problems. The goal is to classify iris flowers into one of three species based on their petal and sepal dimensions.

### Dataset
-**Source**: [Kaggle - IRIS](https://www.kaggle.com/datasets/arshid/iris-flower-dataset)
- **Features**:
  - Sepal Length & Width
  - Petal Length & Width
  - Species (Target variable: Setosa, Versicolor, Virginica)

### Methodology
- Data Exploration & Visualization
- Model Training: K-Nearest Neighbors (KNN), Decision Tree, Random Forest, SVM
- Performance Evaluation (Accuracy, Confusion Matrix, Cross-validation)

### Tools & Libraries
- Python, Pandas, NumPy
- Scikit-Learn, Matplotlib, Seaborn

---
## 3. Credit Card Fraud Detection
### Overview
This project focuses on identifying fraudulent credit card transactions. The dataset consists of real-world anonymized transactions labeled as fraudulent or legitimate.

### Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Features**:
  - Time, Transaction Amount
  - 28 anonymized principal components (PCA transformed features)
  - Fraudulent or Non-Fraudulent (Target variable)

### Methodology
- Data Preprocessing (Handling class imbalance using SMOTE, feature scaling)
- Exploratory Data Analysis (EDA) to understand transaction patterns
- Model Training: Logistic Regression, Random Forest, XGBoost, Neural Networks
- Performance Evaluation (Precision, Recall, ROC-AUC, F1-score)

### Tools & Libraries
- Python, Pandas, NumPy
- Scikit-Learn, XGBoost, Matplotlib, Seaborn
- Imbalanced-learn (for handling class imbalance)

---
## Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ds-projects.git
   cd ds-projects
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run individual project notebooks or scripts:
   ```bash
   python titanic_survival.py
   python iris_classification.py
   python credit_fraud.py
   ```

---
