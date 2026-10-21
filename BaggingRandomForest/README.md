# Breast Cancer Classification using Bagging and Random Forest

## Project Overview
This project demonstrates breast cancer classification using a Bagging Classifier with a Random Forest model. It uses machine learning techniques to classify breast cancer data based on diagnostic features.

## Dataset
- **Dataset File:** `breast_cancer.csv`
- **Total Records:** 569
- **Total Columns:** 31
- **Input Features:** 30
- **Target Column:** `target`

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Machine Learning Pipeline
- Bagging Classifier
- Random Forest Classifier

## Project Workflow
1. Load the dataset using Pandas.
2. Separate input features and target labels.
3. Split the dataset into training and testing sets.
4. Create a Bagging Classification model using Random Forest.
5. Train the model.
6. Predict target values on test data.
7. Evaluate the model using accuracy, classification report, and confusion matrix.

## Model Configuration
- **Ensemble Method:** Bagging Classifier
- **Base Estimator:** Random Forest Classifier
- **Test Size:** 20%
- **Random State:** 42

## Results

The model achieved **92.98% accuracy** on the test dataset.

| Metric | Class 0 | Class 1 |
|---|---:|---:|
| Precision | 0.90 | 0.94 |
| Recall | 0.90 | 0.94 |
| F1-Score | 0.90 | 0.94 |
| Support | 42 | 72 |

- **Accuracy:** 92.98%
- **Test Samples:** 114
- **Correct Predictions:** 106
- **Incorrect Predictions:** 8

### Confusion Matrix

```text
[[38  4]
 [ 4 68]]
```

The model correctly classified 38 samples from Class 0 and 68 samples from Class 1.

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Keep the Python script and `breast_cancer.csv` in the same project folder.

Run:

```bash
python BaggingRandomForestBreastCancer.py
```

## Conclusion
The model achieved 92.98% test accuracy using an ensemble learning approach. This project demonstrates classification, model evaluation, and the use of ensemble methods in machine learning.

**Note:** This project is for educational purposes only and is not intended for clinical diagnosis or medical decision-making.
