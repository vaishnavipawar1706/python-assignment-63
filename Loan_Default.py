import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv("Loan_Default.csv")

print("\n========== DATASET ==========")
print(data)


# ============================================================
# 2. UNDERSTAND THE DATASET
# ============================================================

print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATASET SHAPE ==========")
print(data.shape)

print("\n========== COLUMN NAMES ==========")
print(data.columns)

print("\n========== DATA TYPES ==========")
print(data.dtypes)

print("\n========== STATISTICAL INFORMATION ==========")
print(data.describe())


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())


# ============================================================
# 4. CHECK TARGET CLASS BALANCE
# ============================================================

print("\n========== DEFAULT CLASS COUNT ==========")
print(data["Default"].value_counts())

print("\n========== DEFAULT CLASS PERCENTAGE ==========")
print(data["Default"].value_counts(normalize=True) * 100)


# Plot class distribution

data["Default"].value_counts().plot(
    kind="bar",
    title="Loan Default Class Distribution"
)

plt.xlabel("Default")
plt.ylabel("Number of Applicants")
plt.show()


# ============================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================

# Convert PreviousDefault
data["PreviousDefault"] = data["PreviousDefault"].map({
    "No": 0,
    "Yes": 1
})


# Convert HomeOwnership using one-hot encoding

data = pd.get_dummies(
    data,
    columns=["HomeOwnership"],
    drop_first=True,
    dtype=int
)


print("\n========== DATA AFTER ENCODING ==========")
print(data.head())

print("\n========== NEW COLUMNS ==========")
print(data.columns)


# ============================================================
# 6. SEPARATE X AND y
# ============================================================

X = data.drop("Default", axis=1)

y = data["Default"]


print("\n========== FEATURES (X) ==========")
print(X.head())

print("\n========== TARGET (y) ==========")
print(y.head())


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n========== TRAINING DATA ==========")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n========== TESTING DATA ==========")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ============================================================
# 8. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 9. CREATE MLP CLASSIFIER
# ============================================================

model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)


# ============================================================
# 10. TRAIN MODEL
# ============================================================

model.fit(X_train_scaled, y_train)


print("\n========== MODEL TRAINED SUCCESSFULLY ==========")


# ============================================================
# 11. PREDICTION
# ============================================================

y_pred = model.predict(X_test_scaled)


print("\n========== PREDICTIONS ==========")
print(y_pred)


# ============================================================
# 12. ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== ACCURACY ==========")
print("Accuracy:", accuracy)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\n========== CONFUSION MATRIX ==========")
print(cm)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# 15. PRECISION, RECALL AND F1-SCORE
# ============================================================

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n========== PERFORMANCE METRICS ==========")

print("Precision:", precision)

print("Recall:", recall)

print("F1-Score:", f1)


# ============================================================
# 16. PLOT TRAINING LOSS
# ============================================================

plt.plot(
    model.loss_curve_
)

plt.title("MLP Training Loss")

plt.xlabel("Iterations")

plt.ylabel("Loss")

plt.show()


# ============================================================
# 17. TEST NEW LOAN APPLICANT
# ============================================================

# Example new applicant

new_applicant = pd.DataFrame({
    "Age": [35],
    "Income": [600000],
    "LoanAmount": [400000],
    "CreditScore": [700],
    "EmploymentYears": [8],
    "ExistingLoans": [2],
    "MonthlyDebt": [20000],
    "LoanTerm": [36],
    "PreviousDefault": [0],
    "HomeOwnership_Rent": [1],
    "HomeOwnership_Own": [0]
})


# Make sure columns are in exactly the same order
new_applicant = new_applicant[X.columns]


# Scale new applicant

new_applicant_scaled = scaler.transform(
    new_applicant
)


# Predict

new_prediction = model.predict(
    new_applicant_scaled
)


print("\n========== NEW APPLICANT PREDICTION ==========")

if new_prediction[0] == 0:
    print("Prediction: 0 - Low default risk")
else:
    print("Prediction: 1 - High default risk")


# ============================================================
# HYPERPARAMETER EXPERIMENTS
# ============================================================


# ============================================================
# EXPERIMENT 1 - ACTIVATION FUNCTION
# ============================================================

activations = [
    "identity",
    "logistic",
    "tanh",
    "relu"
]

activation_results = []


for activation in activations:

    temp_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation=activation,
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    temp_model.fit(
        X_train_scaled,
        y_train
    )

    temp_prediction = temp_model.predict(
        X_test_scaled
    )

    temp_accuracy = accuracy_score(
        y_test,
        temp_prediction
    )

    activation_results.append(
        [activation, temp_accuracy]
    )


activation_df = pd.DataFrame(
    activation_results,
    columns=["Activation", "Accuracy"]
)


print("\n========== EXPERIMENT 1: ACTIVATION ==========")
print(activation_df)


# ============================================================
# EXPERIMENT 2 - HIDDEN LAYERS
# ============================================================

hidden_layers = [
    (10,),
    (20, 10),
    (50, 25),
    (100, 50, 25)
]

hidden_results = []


for hidden_layer in hidden_layers:

    temp_model = MLPClassifier(
        hidden_layer_sizes=hidden_layer,
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42
    )

    temp_model.fit(
        X_train_scaled,
        y_train
    )

    temp_prediction = temp_model.predict(
        X_test_scaled
    )

    temp_accuracy = accuracy_score(
        y_test,
        temp_prediction
    )

    hidden_results.append(
        [str(hidden_layer), temp_accuracy]
    )


hidden_df = pd.DataFrame(
    hidden_results,
    columns=["Hidden Layers", "Accuracy"]
)


print("\n========== EXPERIMENT 2: HIDDEN LAYERS ==========")
print(hidden_df)


# ============================================================
# EXPERIMENT 3 - LEARNING RATE
# ============================================================

learning_rates = [
    0.0001,
    0.001,
    0.01,
    0.1
]

learning_results = []


for learning_rate in learning_rates:

    temp_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        learning_rate_init=learning_rate,
        max_iter=1000,
        random_state=42
    )

    temp_model.fit(
        X_train_scaled,
        y_train
    )

    temp_prediction = temp_model.predict(
        X_test_scaled
    )

    temp_accuracy = accuracy_score(
        y_test,
        temp_prediction
    )

    learning_results.append(
        [learning_rate, temp_accuracy]
    )


learning_df = pd.DataFrame(
    learning_results,
    columns=["Learning Rate", "Accuracy"]
)


print("\n========== EXPERIMENT 3: LEARNING RATE ==========")
print(learning_df)
print("Successfully Completed.")
