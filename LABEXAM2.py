# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)

# Load dataset (make sure file name matches the dataset in this folder)
df = pd.read_csv("Lab_Exam_binary_classification_dataset.csv")

# Basic dataset info
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nStatistics:")
print(df.describe())

# Check class balance and missing values
print("\nMissing values (per column):")
print(df.isnull().sum())
print("\nClass counts:")
print(df['Target'].value_counts(dropna=False))

# Drop rows with missing target
df = df.dropna(subset=["Target"]).copy()

# Remove obvious outliers in Feature1 based on the distribution
# (retain values within a reasonable range for modeling)
df = df[df['Feature1'] < 100].copy()

# Create binary target for modeling
df['Target_bin'] = (df['Target'] == 'Yes').astype(int)

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(14, 10))

# Class distribution
plt.subplot(2, 2, 1)
df['Target'].value_counts().plot(kind='bar', color=['blue', 'orange'])
plt.title('Class Distribution')
plt.xlabel('Target')
plt.ylabel('Count')
plt.xticks(rotation=0)

# Feature1 histogram
plt.subplot(2, 2, 2)
sns.histplot(df, x='Feature1', hue='Target', kde=False, palette=['blue', 'orange'])
plt.title('Feature1 Distribution by Class')

# Feature2 histogram
plt.subplot(2, 2, 3)
sns.histplot(df, x='Feature2', hue='Target', kde=False, palette=['blue', 'orange'])
plt.title('Feature2 Distribution by Class')

# Scatter plot
plt.subplot(2, 2, 4)
for label, color in [('No', 'blue'), ('Yes', 'orange')]:
    subset = df[df['Target'] == label]
    plt.scatter(subset['Feature1'], subset['Feature2'], alpha=0.4, label=label, color=color, s=20)
plt.title('Feature1 vs Feature2')
plt.xlabel('Feature1')
plt.ylabel('Feature2')
plt.legend()

plt.tight_layout()
plt.show()

# Build Logistic Regression model
X = df[['Feature1', 'Feature2']].values
y = df['Target_bin'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# Decision boundary plot
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02),
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', alpha=0.7, label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='D', edgecolor='k', alpha=0.9, label='Test')
plt.title('Decision Boundary (scaled features)')
plt.xlabel('Feature1 (scaled)')
plt.ylabel('Feature2 (scaled)')
plt.legend(loc='best')
plt.show()

# Model evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No', 'Yes']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Confusion Matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
plt.colorbar()
plt.show()

# ROC Curve plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc_score(y_test, y_proba):.3f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()
