import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# -----------------------------------------
# Student Depression Prediction Project
# -----------------------------------------

# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

df = pd.read_csv("student_depression_dataset.csv")

print("Original dataset shape:", df.shape)
print(df.head())

TARGET_COL = "Depression"


df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

print("Dataset after cleaning:", df.shape)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL] 

print("Features used:", list(X.columns))
print("Target distribution:")
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\n----------------------------")
print("Model Accuracy:", round(accuracy, 3))
print("----------------------------")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(6,6))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center")

plt.show()

importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

importance["AbsCoefficient"] = importance["Coefficient"].abs()
importance = importance.sort_values("AbsCoefficient", ascending=False)

print("\nTop Features Influencing Depression Prediction:")
print(importance)

plt.figure(figsize=(10,6))
plt.barh(
    importance["Feature"][:10],
    importance["AbsCoefficient"][:10]
)
plt.title("Top 10 Feature Importances")
plt.gca().invert_yaxis()
plt.show()

test_results = X_test.copy()
test_results["Actual"] = y_test.values
test_results["Predicted"] = y_pred

wrong = test_results[test_results["Actual"] != test_results["Predicted"]]

print("\nTotal Misclassified Samples:", len(wrong))

sample_errors = wrong.sample(min(5, len(wrong)), random_state=42)

print("\nFive Misclassified Samples:\n")
print(sample_errors)
