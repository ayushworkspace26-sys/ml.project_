# ml.project_

# 🏠 House Price Prediction & Machine Learning Analysis

A comprehensive Machine Learning project that demonstrates **Regression, Classification, Dimensionality Reduction, and Clustering** using a synthetically generated house-price dataset.

The project applies multiple supervised and unsupervised machine learning algorithms to analyze housing features such as **Area, Number of Rooms, Location Score, and Price**.

---

## 📌 Project Overview

The objective of this project is to build and compare different Machine Learning techniques on a house-price dataset.

The project covers:

* Data generation and preprocessing
* Exploratory data analysis
* Correlation analysis
* Feature scaling
* Regression
* Classification
* Dimensionality reduction
* Clustering
* Model evaluation
* Data visualization

The notebook demonstrates how different Machine Learning algorithms can be applied to the same dataset for different analytical tasks.

---

## 🎯 Objectives

The main objectives of this project are:

1. Generate and analyze a house-price dataset.
2. Study relationships between housing features and price.
3. Predict house prices using **Linear Regression**.
4. Classify houses into price categories using multiple classification algorithms.
5. Compare classification models based on accuracy.
6. Apply **PCA and LDA** for dimensionality reduction.
7. Perform customer/market segmentation using **K-Means Clustering**.
8. Visualize model performance and dataset relationships.

---

## 📊 Dataset

The dataset is generated programmatically inside the Jupyter Notebook using NumPy.

The dataset contains **300 records** and the following features:

| Feature         | Description                                   |
| --------------- | --------------------------------------------- |
| `Area`          | Area of the house                             |
| `Rooms`         | Number of rooms                               |
| `LocationScore` | Numerical score representing location quality |
| `Price`         | Generated house price                         |
| `Category`      | Price category used for classification        |

### Price Generation

The house price is generated using the following relationship:

```text
Price = (Area × 50)
      + (Rooms × 10000)
      + (LocationScore × 5000)
      + Random Noise
```

The generated prices are divided into **3 categories** using `pandas.cut()` for classification.

---

## 🔄 Machine Learning Workflow

```text
Dataset Generation
        ↓
Data Exploration
        ↓
Correlation Analysis
        ↓
Train-Test Split
        ↓
Feature Scaling
        ↓
 ┌───────────────┬────────────────┐
 ↓               ↓                ↓
Regression    Classification   Unsupervised
 ↓               ↓                ↓
Linear       Logistic          K-Means
Regression   Regression
              Decision Tree
              KNN
              Naive Bayes
        ↓
 PCA + LDA
        ↓
 Model Evaluation
        ↓
Final Analysis
```

---

## 🔍 Exploratory Data Analysis

A correlation heatmap is generated to understand relationships between the numerical variables.

One of the main observations from the analysis is that **Area has a strong influence on Price**.

The project uses visualization techniques to understand the generated dataset and model results.

---

# 🤖 Machine Learning Models

## 1. Linear Regression

Linear Regression is used as the primary **regression algorithm** to predict house prices.

### Features

* Area
* Rooms
* Location Score

### Target

* Price

The model is evaluated using:

* Mean Squared Error (MSE)
* R² Score

The notebook also generates an **Actual Price vs Predicted Price** visualization.

---

## 2. Logistic Regression

Logistic Regression is used as a **classification algorithm** to predict the price category of a house.

The model generates:

* Classification accuracy
* Confusion matrix
* ROC curve
* AUC score

---

## 3. Decision Tree

A `DecisionTreeClassifier` is used for price-category classification.

The model is configured with:

```python
max_depth=3
```

The notebook also visualizes the resulting decision tree and displays its actual depth.

---

## 4. K-Nearest Neighbors (KNN)

KNN is used for classification.

The notebook tests different values of `K` from:

```text
K = 1 to 9
```

The accuracy of each value is calculated and visualized.

The best-performing K value is automatically selected based on classification accuracy.

---

## 5. Naive Bayes

A **Gaussian Naive Bayes** classifier is used for price-category classification.

The model performance is evaluated using classification accuracy and a confusion matrix.

---

# 📉 PCA & LDA

The project demonstrates two dimensionality-reduction techniques.

### PCA — Principal Component Analysis

PCA is an **unsupervised dimensionality-reduction technique**.

The notebook reduces the feature space to **2 principal components** and calculates the variance explained by those components.

### LDA — Linear Discriminant Analysis

LDA is a **supervised dimensionality-reduction technique**.

It reduces the feature space to **1 linear discriminant** while attempting to maximize separation between the different price categories.

The notebook visually compares:

```text
PCA → Maximum Variance
LDA → Maximum Class Separation
```

---

# 📦 K-Means Clustering

K-Means is used for **unsupervised learning and market segmentation**.

The notebook initially evaluates different cluster counts from:

```text
K = 2 to 10
```

using:

* Elbow Method
* Silhouette Score

The optimal number of clusters is selected based on the highest silhouette score.

The notebook also performs a separate K-Means visualization for identifying potential **housing market segments**.

Cluster centers are calculated and displayed.

---

# 📈 Model Evaluation

The project evaluates the models using appropriate metrics.

### Regression

| Model             | Evaluation Metrics |
| ----------------- | ------------------ |
| Linear Regression | MSE, R² Score      |

### Classification

| Model               | Evaluation Metric |
| ------------------- | ----------------- |
| Logistic Regression | Accuracy          |
| Decision Tree       | Accuracy          |
| KNN                 | Accuracy          |
| Naive Bayes         | Accuracy          |

### Clustering

| Technique | Evaluation                     |
| --------- | ------------------------------ |
| K-Means   | Elbow Method, Silhouette Score |

### Dimensionality Reduction

| Technique | Analysis           |
| --------- | ------------------ |
| PCA       | Explained Variance |
| LDA       | Class Separation   |

---

## 🛠️ Technologies Used

* **Python**
* **Jupyter Notebook**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

---

## 📚 Python Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans

from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    confusion_matrix,
    r2_score,
    roc_curve,
    auc,
    silhouette_score
)
```

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/house-price-ml-project.git
```

### 2. Navigate to the project directory

```bash
cd house-price-ml-project
```

### 3. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4. Start Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open

```text
mlfinalproject.ipynb
```

### 6. Run all cells

The dataset is generated automatically inside the notebook, so no external dataset is required.

---

# 📁 Project Structure

```text
House-Price-ML-Project/
│
├── mlfinalproject.ipynb
└── README.md
```

---

# 🔬 Key Concepts Demonstrated

This project demonstrates practical implementation of:

* Supervised Learning
* Unsupervised Learning
* Regression
* Classification
* Clustering
* Feature Scaling
* Correlation Analysis
* Confusion Matrix
* ROC Curve
* AUC
* PCA
* LDA
* K-Means
* Elbow Method
* Silhouette Score
* Model Comparison
* Data Visualization

---

# 💡 Project Highlights

### Supervised Learning

The project uses labeled data for:

* Linear Regression
* Logistic Regression
* Decision Tree
* KNN
* Naive Bayes
* LDA

### Unsupervised Learning

The project uses:

* PCA
* K-Means Clustering

to analyze the structure of the dataset without relying solely on target predictions.

---

# 📌 Conclusion

This project successfully demonstrates the application of multiple Machine Learning techniques to a house-price dataset.

The analysis covers the complete workflow from **dataset generation and preprocessing to model training, evaluation, dimensionality reduction, clustering, and visualization**.

It provides a practical comparison between different Machine Learning approaches and demonstrates how regression, classification, and clustering can be used for different types of problems within the same domain.

---

## 👨‍💻 Author

**Ayush Uniyal**

Machine Learning Project

---

## ⭐ Future Improvements

Possible improvements include:

* Use a real-world house-price dataset.
* Add Random Forest and Gradient Boosting models.
* Perform hyperparameter tuning.
* Add cross-validation.
* Add feature engineering.
* Build an interactive Streamlit dashboard.
* Deploy the model as a web application.
* Compare additional regression and classification algorithms.
