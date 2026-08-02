import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# ==========================
# BASIC INFORMATION
# ==========================

print("First 5 Rows")
print(df.head())

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns)

print("\nInfo")
df.info()

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Values")
print(df.duplicated().sum())

print("\nStatistical Summary")
print(df.describe())

# ==========================
# DATA VISUALIZATION
# ==========================

# Attrition Count
plt.figure(figsize=(6,4))
df["Attrition"].value_counts().plot(kind="bar")
plt.title("Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Age Distribution
plt.figure(figsize=(8,5))
plt.hist(df["Age"], bins=15)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Employees")
plt.tight_layout()
plt.show()

# Department-wise Attrition
pd.crosstab(df["Department"], df["Attrition"]).plot(
    kind="bar",
    figsize=(8,5)
)
plt.title("Department-wise Attrition")
plt.tight_layout()
plt.show()

# Gender-wise Attrition
pd.crosstab(df["Gender"], df["Attrition"]).plot(
    kind="bar",
    figsize=(6,4)
)
plt.title("Gender-wise Attrition")
plt.tight_layout()
plt.show()

# Job Role-wise Attrition
pd.crosstab(df["JobRole"], df["Attrition"]).plot(
    kind="bar",
    figsize=(12,5)
)
plt.title("Job Role-wise Attrition")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Monthly Income Distribution
plt.figure(figsize=(8,5))
plt.hist(df["MonthlyIncome"], bins=20)
plt.title("Monthly Income Distribution")
plt.xlabel("Monthly Income")
plt.ylabel("Employees")
plt.tight_layout()
plt.show()

print("\nEDA Completed Successfully")

# ==========================
# MACHINE LEARNING IMPORTS
# ==========================

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import seaborn as sns
# ==========================
# DATA PREPROCESSING
# ==========================

# Copy Dataset
data = df.copy()

# Label Encoding
le = LabelEncoder()

for column in data.columns:
    if data[column].dtype == "object":
        data[column] = le.fit_transform(data[column])

# Features and Target
X = data.drop("Attrition", axis=1)
y = data["Attrition"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# LOGISTIC REGRESSION
# ==========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train_scaled, y_train)

lr_pred = lr_model.predict(X_test_scaled)

print("\n========== Logistic Regression ==========")
print("Accuracy :", accuracy_score(y_test, lr_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report")
print(classification_report(y_test, lr_pred))

# ==========================
# RANDOM FOREST
# ==========================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n========== Random Forest ==========")
print("Accuracy :", accuracy_score(y_test, rf_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report")
print(classification_report(y_test, rf_pred))

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

importance = importance.sort_values(
    ascending=False
)

print("\nTop 10 Important Features")
print(importance.head(10))

plt.figure(figsize=(10,6))
importance.head(10).plot(kind="bar")

plt.title("Top 10 Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("feature_importance.png", dpi=300)

plt.show()
# ==========================
# SAVE CLEANED DATASET
# ==========================

data.to_csv("employee_turnover_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")

# ==========================
# CONFUSION MATRIX (Random Forest)
# ==========================

import seaborn as sns

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

# ==========================
# PROJECT COMPLETED
# ==========================

print("\n========================================")
print("Employee Turnover Project Completed")
print("========================================")
print("Files Generated:")
print("1. employee_turnover_cleaned.csv")
print("2. confusion_matrix.png")
print("3. feature_importance.png")
print("========================================")