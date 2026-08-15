import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("train.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nSurvival rate:")
print(df["Survived"].value_counts())
# Drop columns we won't use
df = df.drop(columns=["Cabin", "Name", "Ticket", "PassengerId"])

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with most common value (mode)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Convert Sex to numbers: male=0, female=1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Convert Embarked to numbers (one-hot encoding)
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

print(df.head())
print("\nMissing values now:")
print(df.isnull().sum())
print("\nColumns now:", df.columns.tolist())
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Features and label
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (needed for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

results = {}

# 1. Logistic Regression
model1 = LogisticRegression(max_iter=1000)
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)
results["Logistic Regression"] = accuracy_score(y_test, pred1)

# 2. KNN (uses scaled data)
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train_scaled, y_train)
pred2 = model2.predict(X_test_scaled)
results["KNN"] = accuracy_score(y_test, pred2)

# 3. Decision Tree
model3 = DecisionTreeClassifier(max_depth=5, random_state=42)
model3.fit(X_train, y_train)
pred3 = model3.predict(X_test)
results["Decision Tree"] = accuracy_score(y_test, pred3)

# 4. Naive Bayes
model4 = GaussianNB()
model4.fit(X_train, y_train)
pred4 = model4.predict(X_test)
results["Naive Bayes"] = accuracy_score(y_test, pred4)

# Compare all
print("\n--- Model Comparison ---")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} with accuracy {results[best_model]:.4f}")