
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Step 1: Load Dataset
BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "breast_cancer.csv")

print("Shape of dataset:", df.shape)
print("\nFirst 5 records:")
print(df.head())

# Step 2: Separate Features and Target
X = df.drop("target", axis=1)
Y = df["target"]

# Step 3: Split Dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

# Step 4: Create Machine Learning Pipeline
model = Pipeline([
    ("classifier", BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=42),
        n_estimators=10,
        random_state=42
    ))
])

# Step 5: Train Model
model.fit(X_train, Y_train)

# Step 6: Make Predictions
Y_pred = model.predict(X_test)

# Step 7: Evaluate Model
print("\nBagging Model Accuracy:",
      round(accuracy_score(Y_test, Y_pred) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# Step 8: Plot and Save Confusion Matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

display.plot(cmap="Blues")
plt.title("Breast Cancer Bagging Classification")
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png")
plt.show()