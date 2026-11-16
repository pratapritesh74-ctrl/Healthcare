# ============================================================
# PHARMEASY HEALTHCARE DECISION INTELLIGENCE
# PART 2: LOGISTIC REGRESSION, KNN AND COST-BASED EVALUATION
# TASKS 2.1 TO 2.8
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import csv
import os

# ------------------------------------------------------------
# PROJECT SETUP
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# TASK 2.1: STRATIFIED 70/30 TRAIN-TEST SPLIT
# ------------------------------------------------------------

df = pd.read_csv(BASE_DIR / "data" / "records.csv")

feature_columns = [
    "months_registered",
    "missed_visit_count_12m",
    "copay_burden_ratio",
    "avg_bill_inr_thousands",
    "adherence_score",
    "chronic_condition_flag",
    "insured_flag"
]

target_column = "no_show"

X = df[feature_columns].to_numpy(dtype=float)
y = df[target_column].to_numpy(dtype=int)

# Separate positive and negative class indices
positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]

# First floor(70%) of each class goes to training
positive_split = (7 * len(positive_indices)) // 10
negative_split = (7 * len(negative_indices)) // 10

train_indices = np.sort(np.concatenate([
    positive_indices[:positive_split],
    negative_indices[:negative_split]
]))

test_indices = np.sort(np.concatenate([
    positive_indices[positive_split:],
    negative_indices[negative_split:]
]))

X_train = X[train_indices]
y_train = y[train_indices]

X_test = X[test_indices]
y_test = y[test_indices]

# Standardise using training mean and sample standard deviation
train_means = np.mean(X_train, axis=0)
train_stds = np.std(X_train, axis=0, ddof=1)

X_train_std = (X_train - train_means) / train_stds
X_test_std = (X_test - train_means) / train_stds

print("\nTASK 2.1: STRATIFIED SPLIT")
print("train_rows:", len(y_train))
print("train_positives:", np.sum(y_train == 1))
print("test_rows:", len(y_test))
print("test_positives:", np.sum(y_test == 1))


# ------------------------------------------------------------
# COMMON FUNCTIONS
# ------------------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def binary_cross_entropy(y_true, probabilities):

    probabilities = np.clip(
        probabilities, 1e-15, 1 - 1e-15
    )

    return -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )


# ------------------------------------------------------------
# TASK 2.2: LOGISTIC REGRESSION FROM SCRATCH
# ------------------------------------------------------------

weights = np.zeros(X_train_std.shape[1])
bias = 0.0

learning_rate = 0.1
epochs = 1000

loss_history = {}

# Initial loss before updates
initial_probabilities = sigmoid(
    X_train_std @ weights + bias
)

loss_history[0] = binary_cross_entropy(
    y_train, initial_probabilities
)

for epoch in range(1, epochs + 1):

    z = X_train_std @ weights + bias
    probabilities = sigmoid(z)

    error = probabilities - y_train

    dw = (X_train_std.T @ error) / len(y_train)
    db = np.mean(error)

    weights = weights - learning_rate * dw
    bias = bias - learning_rate * db

    if epoch in [1, 10, 100, 1000]:

        updated_probabilities = sigmoid(
            X_train_std @ weights + bias
        )

        loss_history[epoch] = binary_cross_entropy(
            y_train, updated_probabilities
        )

# Training and test probabilities
train_probabilities = sigmoid(
    X_train_std @ weights + bias
)

test_probabilities = sigmoid(
    X_test_std @ weights + bias
)

test_bce = binary_cross_entropy(
    y_test, test_probabilities
)

print("\nTASK 2.2: LOGISTIC REGRESSION")

for epoch in [0, 1, 10, 100, 1000]:
    print(f"loss_epoch_{epoch}:", loss_history[epoch])

for i, weight in enumerate(weights, start=1):
    print(f"weight_X{i}:", weight)

print("bias:", bias)
print("test_bce:", test_bce)


# ------------------------------------------------------------
# TASK 2.3: LEARNING-RATE COMPARISON
# ------------------------------------------------------------

learning_rates = [0.01, 0.1, 1.0]
learning_rate_losses = {}

for eta in learning_rates:

    # Restart from zero for every learning rate
    w = np.zeros(X_train_std.shape[1])
    b = 0.0

    # Run 200 full-batch updates
    for epoch in range(200):

        z = X_train_std @ w + b
        probabilities = sigmoid(z)

        error = probabilities - y_train

        dw = (X_train_std.T @ error) / len(y_train)
        db = np.mean(error)

        w = w - eta * dw
        b = b - eta * db

    # Calculate final training loss
    final_probabilities = sigmoid(
        X_train_std @ w + b
    )

    final_loss = binary_cross_entropy(
        y_train, final_probabilities
    )

    learning_rate_losses[eta] = final_loss

print("\nTASK 2.3: LEARNING-RATE COMPARISON")

for eta in learning_rates:
    print(
        f"loss_epoch_200_eta_{eta}:",
        learning_rate_losses[eta]
    )


# ------------------------------------------------------------
# TASK 2.4: ODDS RATIOS
# ------------------------------------------------------------

odds_ratios = np.exp(weights)

print("\nTASK 2.4: ODDS RATIOS")

for i, odds_ratio in enumerate(odds_ratios, start=1):
    print(f"odds_ratio_X{i}:", odds_ratio)


# ------------------------------------------------------------
# TASK 2.5: MODEL EVALUATION AT TWO THRESHOLDS
# ------------------------------------------------------------

def evaluate_model(y_true, probabilities, threshold):

    predictions = (probabilities >= threshold).astype(int)

    TP = int(np.sum((y_true == 1) & (predictions == 1)))
    TN = int(np.sum((y_true == 0) & (predictions == 0)))
    FP = int(np.sum((y_true == 0) & (predictions == 1)))
    FN = int(np.sum((y_true == 1) & (predictions == 0)))

    accuracy = (TP + TN) / len(y_true)

    precision = (
        TP / (TP + FP) if TP + FP > 0 else 0
    )

    recall = (
        TP / (TP + FN) if TP + FN > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    specificity = (
        TN / (TN + FP) if TN + FP > 0 else 0
    )

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }


print("\nTASK 2.5: MODEL EVALUATION")

threshold_metrics = {}

for threshold in [0.5, 0.3]:

    metrics = evaluate_model(
        y_test,
        test_probabilities,
        threshold
    )

    threshold_metrics[threshold] = metrics

    print(f"\nThreshold: {threshold}")

    for key, value in metrics.items():
        print(f"{key}: {value}")


blind_accuracy = np.mean(y_test == 0)

print("\nblind_accuracy:", blind_accuracy)


# ------------------------------------------------------------
# TASK 2.6: THRESHOLD SWEEP
# ------------------------------------------------------------

threshold_rows = []

for threshold in np.arange(0.1, 1.0, 0.1):

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    TP = int(np.sum((y_test == 1) & (predictions == 1)))
    FP = int(np.sum((y_test == 0) & (predictions == 1)))
    TN = int(np.sum((y_test == 0) & (predictions == 0)))
    FN = int(np.sum((y_test == 1) & (predictions == 0)))

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)

    threshold_rows.append([
        round(float(threshold), 1),
        TP, FP, TN, FN, TPR, FPR
    ])

with open(
    OUTPUT_DIR / "part2_thresholds.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "tau", "tp", "fp", "tn", "fn", "tpr", "fpr"
    ])

    writer.writerows([
        [
            f"{row[0]:.4f}",
            int(row[1]),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            f"{row[5]:.4f}",
            f"{row[6]:.4f}"
        ]
        for row in threshold_rows
    ])

print("\nTASK 2.6: THRESHOLD SWEEP")

for row in threshold_rows:
    print(row)

print("Threshold CSV saved successfully!")


# ------------------------------------------------------------
# TASK 2.7: KNN WITH 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

def knn_predict(X_reference, y_reference, X_query, K):

    predictions = []

    for query in X_query:

        distances = np.sqrt(
            np.sum((X_reference - query) ** 2, axis=1)
        )

        # Stable sorting preserves reference-row order for ties
        nearest_indices = np.argsort(
            distances, kind="stable"
        )[:K]

        nearest_labels = y_reference[nearest_indices]

        prediction = int(
            np.sum(nearest_labels) >= (K + 1) / 2
        )

        predictions.append(prediction)

    return np.array(predictions)


def calculate_f1(y_true, y_pred):

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = (
        TP / (TP + FP) if TP + FP > 0 else 0
    )

    recall = (
        TP / (TP + FN) if TP + FN > 0 else 0
    )

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )


knn_results = []

for K in range(1, 22, 2):

    fold_f1_scores = []

    for fold in range(5):

        validation_indices = (
            np.arange(len(y_train)) % 5 == fold
        )

        reference_indices = ~validation_indices

        X_reference = X_train_std[reference_indices]
        y_reference = y_train[reference_indices]

        X_validation = X_train_std[validation_indices]
        y_validation = y_train[validation_indices]

        predictions = knn_predict(
            X_reference,
            y_reference,
            X_validation,
            K
        )

        fold_f1 = calculate_f1(
            y_validation,
            predictions
        )

        fold_f1_scores.append(fold_f1)

    mean_f1 = np.mean(fold_f1_scores)

    knn_results.append([K, mean_f1])


# Highest mean F1; ties go to smaller K
knn_best_k = max(
    knn_results,
    key=lambda row: (row[1], -row[0])
)[0]

knn_test_predictions = knn_predict(
    X_train_std,
    y_train,
    X_test_std,
    knn_best_k
)

knn_tp = int(np.sum(
    (y_test == 1) & (knn_test_predictions == 1)
))

knn_fp = int(np.sum(
    (y_test == 0) & (knn_test_predictions == 1)
))

knn_tn = int(np.sum(
    (y_test == 0) & (knn_test_predictions == 0)
))

knn_fn = int(np.sum(
    (y_test == 1) & (knn_test_predictions == 0)
))

knn_accuracy = (
    (knn_tp + knn_tn) / len(y_test)
)

knn_precision = (
    knn_tp / (knn_tp + knn_fp)
    if knn_tp + knn_fp > 0 else 0
)

knn_recall = (
    knn_tp / (knn_tp + knn_fn)
    if knn_tp + knn_fn > 0 else 0
)

knn_f1 = calculate_f1(
    y_test,
    knn_test_predictions
)

with open(
    OUTPUT_DIR / "part2_knn_cv.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(["K", "mean_cv_f1"])

    writer.writerows([
        [int(row[0]), f"{row[1]:.4f}"]
        for row in knn_results
    ])

print("\nTASK 2.7: KNN CROSS-VALIDATION")

for row in knn_results:
    print(row)

print("knn_best_k:", knn_best_k)
print("knn_tp:", knn_tp)
print("knn_fp:", knn_fp)
print("knn_tn:", knn_tn)
print("knn_fn:", knn_fn)
print("knn_accuracy:", knn_accuracy)
print("knn_precision:", knn_precision)
print("knn_recall:", knn_recall)
print("knn_f1:", knn_f1)

print("KNN CSV saved successfully!")


# ------------------------------------------------------------
# TASK 2.8: COST-BASED THRESHOLD SELECTION
# ------------------------------------------------------------

cost_results = []

FP_COST = 50
FN_COST = 400

for k in range(1, 20):

    threshold = k / 20

    train_predictions = (
        train_probabilities >= threshold
    ).astype(int)

    FP = int(np.sum(
        (y_train == 0) & (train_predictions == 1)
    ))

    FN = int(np.sum(
        (y_train == 1) & (train_predictions == 0)
    ))

    total_cost = FP_COST * FP + FN_COST * FN

    cost_results.append([
        threshold, FP, FN, total_cost
    ])


# Minimum training cost; ties go to lower threshold
best_cost_row = min(
    cost_results,
    key=lambda row: (row[3], row[0])
)

cost_tau = best_cost_row[0]
cost_train_at_tau = best_cost_row[3]

test_predictions_cost = (
    test_probabilities >= cost_tau
).astype(int)

test_FP = int(np.sum(
    (y_test == 0) & (test_predictions_cost == 1)
))

test_FN = int(np.sum(
    (y_test == 1) & (test_predictions_cost == 0)
))

cost_test_at_tau = (
    FP_COST * test_FP + FN_COST * test_FN
)

test_predictions_05 = (
    test_probabilities >= 0.5
).astype(int)

fp_05 = int(np.sum(
    (y_test == 0) & (test_predictions_05 == 1)
))

fn_05 = int(np.sum(
    (y_test == 1) & (test_predictions_05 == 0)
))

cost_test_at_0_5 = (
    FP_COST * fp_05 + FN_COST * fn_05
)

cost_test_flag_nobody = (
    FN_COST * int(np.sum(y_test == 1))
)

print("\nTASK 2.8: COST-BASED THRESHOLD SELECTION")

for row in cost_results:
    print(row)

print("cost_tau:", cost_tau)
print("cost_train_at_tau:", cost_train_at_tau)
print("cost_test_at_tau:", cost_test_at_tau)
print("cost_test_at_0.5:", cost_test_at_0_5)
print("cost_test_flag_nobody:", cost_test_flag_nobody)


# ------------------------------------------------------------
# SAVE ALL PART 2 RESULTS
# ------------------------------------------------------------

results = {}

# Task 2.1
results["train_rows"] = len(y_train)
results["train_positives"] = int(np.sum(y_train == 1))
results["test_rows"] = len(y_test)
results["test_positives"] = int(np.sum(y_test == 1))

# Task 2.2
for epoch in [0, 1, 10, 100, 1000]:
    results[f"loss_epoch_{epoch}"] = loss_history[epoch]

for i, weight in enumerate(weights, start=1):
    results[f"weight_X{i}"] = weight

results["bias"] = bias
results["test_bce"] = test_bce

# Task 2.3
for eta in learning_rates:
    results[f"loss_epoch_200_eta_{eta}"] = (
        learning_rate_losses[eta]
    )

# Task 2.4
for i, odds_ratio in enumerate(odds_ratios, start=1):
    results[f"odds_ratio_X{i}"] = odds_ratio

# Task 2.5
for threshold in [0.5, 0.3]:

    metrics = threshold_metrics[threshold]
    prefix = f"tau_{threshold}_"

    for key, value in metrics.items():
        results[prefix + key.lower()] = value

results["blind_accuracy"] = blind_accuracy

# Task 2.7
results["knn_best_k"] = int(knn_best_k)
results["knn_tp"] = knn_tp
results["knn_fp"] = knn_fp
results["knn_tn"] = knn_tn
results["knn_fn"] = knn_fn
results["knn_accuracy"] = knn_accuracy
results["knn_precision"] = knn_precision
results["knn_recall"] = knn_recall
results["knn_f1"] = knn_f1

# Task 2.8
results["cost_tau"] = cost_tau
results["cost_train_at_tau"] = cost_train_at_tau
results["cost_test_at_tau"] = cost_test_at_tau
results["cost_test_at_0.5"] = cost_test_at_0_5
results["cost_test_flag_nobody"] = cost_test_flag_nobody


# Save results in required key = value format
with open(
    OUTPUT_DIR / "part2_results.txt",
    "w"
) as file:

    for key, value in results.items():

        if isinstance(value, (int, np.integer)):
            formatted_value = str(int(value))
        else:
            formatted_value = f"{float(value):.4f}"

        file.write(f"{key} = {formatted_value}\n")


# ------------------------------------------------------------
# FINAL VERIFICATION
# ------------------------------------------------------------

print("\n============================================")
print("PART 2 COMPLETED")
print("============================================")

print("Total result keys:", len(results))
print("Expected result keys: 61")

print("\nGenerated files:")

for filename in [
    "part2_results.txt",
    "part2_thresholds.csv",
    "part2_knn_cv.csv"
]:

    path = OUTPUT_DIR / filename

    print(filename, ":", path.exists())

print("\nAll Part 2 tasks completed.")
