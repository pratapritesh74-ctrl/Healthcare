from pathlib import Path
import csv
import numpy as np
import math

# ============================================================
# PHARMEASY HEALTHCARE DECISION INTELLIGENCE
# PART 4: MODEL CHOICE AND ARCHITECTURE MEMO
# TASKS 4.1 TO 4.5
# ============================================================

BASE_DIR = Path("/content/Healthcare")
DATA_FILE = BASE_DIR / "data" / "records.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

with open(DATA_FILE, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

feature_columns = [
    "months_registered",
    "missed_visit_count_12m",
    "copay_burden_ratio",
    "avg_bill_inr_thousands",
    "adherence_score",
    "chronic_condition_flag",
    "insured_flag"
]

X = np.array([
    [float(row[col]) for col in feature_columns]
    for row in rows
])

y = np.array([int(row["no_show"]) for row in rows])

# ------------------------------------------------------------
# RECREATE EXACT PART 2 TRAIN-TEST SPLIT
# ------------------------------------------------------------

positive_indices = np.where(y == 1)[0]
negative_indices = np.where(y == 0)[0]

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

train_means = np.mean(X_train, axis=0)
train_stds = np.std(X_train, axis=0, ddof=1)

X_train_std = (X_train - train_means) / train_stds
X_test_std = (X_test - train_means) / train_stds

assert len(y_train) == 560
assert len(y_test) == 240
assert np.sum(y_train == 1) == 119
assert np.sum(y_test == 1) == 51

print("Part 2 split successfully recreated.")

# ------------------------------------------------------------
# RECREATE PART 2 LOGISTIC REGRESSION
# ------------------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

weights = np.zeros(X_train_std.shape[1])
bias = 0.0

learning_rate = 0.1

for epoch in range(1000):
    probabilities = sigmoid(X_train_std @ weights + bias)
    error = probabilities - y_train

    dw = (X_train_std.T @ error) / len(y_train)
    db = np.mean(error)

    weights -= learning_rate * dw
    bias -= learning_rate * db

logistic_train_prob = sigmoid(
    X_train_std @ weights + bias
)

logistic_test_prob = sigmoid(
    X_test_std @ weights + bias
)

# ------------------------------------------------------------
# RECREATE PART 2 KNN AND CROSS-VALIDATION
# ------------------------------------------------------------

def knn_probabilities(X_reference, y_reference, X_query, K):

    probabilities = []

    for query in X_query:
        distances = np.sqrt(
            np.sum((X_reference - query) ** 2, axis=1)
        )

        nearest_indices = np.argsort(
            distances, kind="stable"
        )[:K]

        nearest_labels = y_reference[nearest_indices]

        probabilities.append(np.mean(nearest_labels))

    return np.array(probabilities)


def calculate_f1(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )


knn_cv_results = []

for K in range(1, 22, 2):

    fold_scores = []

    for fold in range(5):

        validation_mask = (
            np.arange(len(y_train)) % 5 == fold
        )

        reference_mask = ~validation_mask

        probabilities = knn_probabilities(
            X_train_std[reference_mask],
            y_train[reference_mask],
            X_train_std[validation_mask],
            K
        )

        predictions = (
            probabilities >= (K + 1) / (2 * K)
        ).astype(int)

        fold_scores.append(
            calculate_f1(
                y_train[validation_mask],
                predictions
            )
        )

    knn_cv_results.append([K, np.mean(fold_scores)])

knn_best_k = max(
    knn_cv_results,
    key=lambda item: (item[1], -item[0])
)[0]

knn_test_prob = knn_probabilities(
    X_train_std,
    y_train,
    X_test_std,
    knn_best_k
)

knn_test_pred = (
    knn_test_prob >= (knn_best_k + 1) / (2 * knn_best_k)
).astype(int)

print("KNN best K:", knn_best_k)

# ------------------------------------------------------------
# TASK 4.1: ENTROPY AND GINI SPLITS
# ------------------------------------------------------------

def entropy(labels):

    n = len(labels)

    if n == 0:
        return 0.0

    positive = np.sum(labels == 1)
    negative = n - positive

    result = 0.0

    for count in [positive, negative]:
        if count > 0:
            p = count / n
            result -= p * math.log2(p)

    return result


def gini(labels):

    n = len(labels)

    if n == 0:
        return 0.0

    positive = np.sum(labels == 1) / n
    negative = 1 - positive

    return 1 - positive**2 - negative**2


root_entropy = entropy(y_train)
root_gini = gini(y_train)

split_conditions = {
    "S1": X_train[:, 5] == 1,
    "S2": X_train[:, 6] == 1,
    "S3": X_train[:, 2] > 0.5,
    "S4": X_train[:, 0] <= 24
}

split_rows = []

for name, condition in split_conditions.items():

    yes_labels = y_train[condition]
    no_labels = y_train[~condition]

    n_yes = len(yes_labels)
    n_no = len(no_labels)

    positives_yes = int(np.sum(yes_labels == 1))
    positives_no = int(np.sum(no_labels == 1))

    weighted_entropy = (
        (n_yes / len(y_train)) * entropy(yes_labels)
        + (n_no / len(y_train)) * entropy(no_labels)
    )

    weighted_gini = (
        (n_yes / len(y_train)) * gini(yes_labels)
        + (n_no / len(y_train)) * gini(no_labels)
    )

    info_gain = root_entropy - weighted_entropy
    gini_gain = root_gini - weighted_gini

    split_rows.append([
        name,
        n_yes,
        positives_yes,
        n_no,
        positives_no,
        info_gain,
        gini_gain
    ])

best_info = max(split_rows, key=lambda r: r[5])[0]
best_gini = max(split_rows, key=lambda r: r[6])[0]

with open(OUTPUT_DIR / "part4_splits.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "split", "n_yes", "positives_yes",
        "n_no", "positives_no",
        "info_gain", "gini_gain"
    ])

    for r in split_rows:
        writer.writerow([
            r[0], r[1], r[2], r[3], r[4],
            f"{r[5]:.4f}", f"{r[6]:.4f}"
        ])

print("\nTASK 4.1: SPLIT RESULTS")
print("root_entropy:", root_entropy)
print("root_gini:", root_gini)
print("best_split_by_info_gain:", best_info)
print("best_split_by_gini_gain:", best_gini)

# ------------------------------------------------------------
# TASK 4.2: DECISION STUMP AND VOTING
# ------------------------------------------------------------

best_condition = split_conditions[best_info]

stump_rate_yes = np.mean(y_train[best_condition])
stump_rate_no = np.mean(y_train[~best_condition])

stump_test_prob = np.where(
    best_condition=None,
    x=0,
    y=0
) if False else np.where(
    {
        "S1": X_test[:, 5] == 1,
        "S2": X_test[:, 6] == 1,
        "S3": X_test[:, 2] > 0.5,
        "S4": X_test[:, 0] <= 24
    }[best_info],
    stump_rate_yes,
    stump_rate_no
)

stump_test_pred = (
    stump_test_prob >= 0.3
).astype(int)

logistic_test_pred = (
    logistic_test_prob >= 0.3
).astype(int)

hard_vote = (
    logistic_test_pred
    + knn_test_pred
    + stump_test_pred
)

hard_vote_pred = (hard_vote >= 2).astype(int)

soft_vote_prob = (
    logistic_test_prob
    + knn_test_prob
    + stump_test_prob
) / 3

soft_vote_pred = (
    soft_vote_prob >= 0.3
).astype(int)


def confusion_metrics(y_true, predictions):

    tp = int(np.sum((y_true == 1) & (predictions == 1)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return tp, fp, tn, fn, precision, recall, f1


voting_models = {
    "logistic": logistic_test_pred,
    "knn": knn_test_pred,
    "stump": stump_test_pred,
    "hard_vote": hard_vote_pred,
    "soft_vote": soft_vote_pred
}

voting_rows = []

for name, predictions in voting_models.items():

    metrics = confusion_metrics(y_test, predictions)

    voting_rows.append([name] + list(metrics))

with open(OUTPUT_DIR / "part4_voting.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "model", "tp", "fp", "tn", "fn",
        "precision", "recall", "f1"
    ])

    for r in voting_rows:
        writer.writerow(
            r[:5] + [f"{v:.4f}" for v in r[5:]]
        )

print("\nTASK 4.2: VOTING RESULTS")

for r in voting_rows:
    print(r)

# ------------------------------------------------------------
# TASK 4.3: TWO ROUNDS OF ADABOOST
# ------------------------------------------------------------

true_y = np.array([
    1, 1, 1, -1, -1, -1, 1, -1
])

h1 = np.array([
    1, 1, -1, -1, -1, -1, -1, -1
])

h2 = np.array([
    -1, 1, 1, -1, 1, -1, 1, -1
])

weights_before = np.ones(8) / 8

adaboost_rows = []

round_results = []

for round_number, weak_predictions in [
    (1, h1),
    (2, h2)
]:

    error_mask = weak_predictions != true_y

    weighted_error = np.sum(
        weights_before[error_mask]
    )

    alpha = 0.5 * np.log(
        (1 - weighted_error) / weighted_error
    )

    unnormalised = (
        weights_before
        * np.exp(-alpha * true_y * weak_predictions)
    )

    weights_after = (
        unnormalised / np.sum(unnormalised)
    )

    round_results.append([weighted_error, alpha])

    for i in range(8):

        adaboost_rows.append([
            round_number,
            i + 1,
            weights_before[i],
            unnormalised[i],
            weights_after[i]
        ])

    weights_before = weights_after.copy()


round1_error, round1_alpha = round_results[0]
round2_error, round2_alpha = round_results[1]

ensemble_score = (
    round1_alpha * h1
    + round2_alpha * h2
)

ensemble_predictions = np.where(
    ensemble_score >= 0, 1, -1
)

ensemble_accuracy = np.mean(
    ensemble_predictions == true_y
)

with open(OUTPUT_DIR / "part4_adaboost.csv", "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "round", "point", "weight_before",
        "weight_unnormalised", "weight_after"
    ])

    for r in adaboost_rows:
        writer.writerow([
            r[0], r[1],
            f"{r[2]:.4f}",
            f"{r[3]:.4f}",
            f"{r[4]:.4f}"
        ])

print("\nTASK 4.3: ADABOOST")
print("round1_error:", round1_error)
print("round1_alpha:", round1_alpha)
print("round2_error:", round2_error)
print("round2_alpha:", round2_alpha)
print("ensemble_accuracy:", ensemble_accuracy)

# ------------------------------------------------------------
# TASK 4.4: OUT-OF-BAG FRACTION
# ------------------------------------------------------------

oob_fraction = (1 - 1 / 560) ** 560

print("\nTASK 4.4: OUT-OF-BAG")
print("oob_fraction:", oob_fraction)

# ------------------------------------------------------------
# TASK 4.5: NEURAL NETWORK PARAMETER COUNTS
# ------------------------------------------------------------

# MLP: 784 -> 16 -> 16 -> 10

mlp_hidden1 = 784 * 16 + 16
mlp_hidden2 = 16 * 16 + 16
mlp_output = 16 * 10 + 10

mlp_total = (
    mlp_hidden1 + mlp_hidden2 + mlp_output
)

# CNN:
# Input: 28 x 28 x 1
# Conv1: 5 x 5, 6 filters, stride 1, padding 2
# Pool1: 2 x 2, stride 2
# Conv2: 5 x 5, 16 filters, stride 1, padding 0
# Pool2: 2 x 2, stride 2
# Flatten -> Dense 32 -> Output 10

cnn_conv1 = (5 * 5 * 1 + 1) * 6

# Conv1 output: 28 x 28 x 6
# Pool1 output: 14 x 14 x 6
# Conv2 output: 10 x 10 x 16
# Pool2 output: 5 x 5 x 16
# Flatten: 400

cnn_conv2 = (5 * 5 * 6 + 1) * 16

flattened_size = 5 * 5 * 16

cnn_dense1 = flattened_size * 32 + 32
cnn_dense2 = 32 * 10 + 10

cnn_total = (
    cnn_conv1 + cnn_conv2
    + cnn_dense1 + cnn_dense2
)

# Fully connected vs convolutional layer

fc_layer_params = (128 * 128 * 3) * 1024 + 1024

conv_layer_params = (3 * 3 * 3) * 64 + 64

# RNN:
# Input dimension = 64
# Hidden dimension = 128
# Output dimension = 16

W_xh = 128 * 64
W_hh = 128 * 128
b_h = 128
W_hy = 16 * 128
b_y = 16

rnn_total = W_xh + W_hh + b_h + W_hy + b_y

architecture_rows = [
    ["mlp_hidden1", mlp_hidden1],
    ["mlp_hidden2", mlp_hidden2],
    ["mlp_output", mlp_output],
    ["cnn_conv1", cnn_conv1],
    ["cnn_conv2", cnn_conv2],
    ["cnn_dense1", cnn_dense1],
    ["cnn_dense2", cnn_dense2]
]

with open(
    OUTPUT_DIR / "part4_architecture.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)
    writer.writerow(["layer", "parameters"])
    writer.writerows(architecture_rows)

print("\nTASK 4.5: ARCHITECTURE")
print("mlp_total:", mlp_total)
print("cnn_total:", cnn_total)
print("fc_layer_params:", fc_layer_params)
print("conv_layer_params:", conv_layer_params)
print("rnn_total:", rnn_total)

# ------------------------------------------------------------
# SAVE PART 4 RESULTS
# ------------------------------------------------------------

results = {
    "root_entropy": root_entropy,
    "root_gini": root_gini,
    "best_split_by_info_gain": best_info,
    "best_split_by_gini_gain": best_gini,
    "stump_rate_yes": stump_rate_yes,
    "stump_rate_no": stump_rate_no,
    "round1_error": round1_error,
    "round1_alpha": round1_alpha,
    "round2_error": round2_error,
    "round2_alpha": round2_alpha,
    "ensemble_accuracy": ensemble_accuracy,
    "oob_fraction": oob_fraction,
    "mlp_total": mlp_total,
    "cnn_total": cnn_total,
    "fc_layer_params": fc_layer_params,
    "conv_layer_params": conv_layer_params,
    "rnn_total": rnn_total
}

with open(
    OUTPUT_DIR / "part4_results.txt",
    "w"
) as f:

    for key, value in results.items():

        if isinstance(value, (int, np.integer)):
            formatted = str(int(value))

        elif isinstance(value, str):
            formatted = value

        else:
            formatted = f"{float(value):.4f}"

        f.write(f"{key} = {formatted}\n")

# ------------------------------------------------------------
# FINAL VERIFICATION
# ------------------------------------------------------------

print("\n============================================")
print("PART 4 COMPLETED")
print("============================================")

expected_files = [
    "part4_results.txt",
    "part4_splits.csv",
    "part4_voting.csv",
    "part4_adaboost.csv",
    "part4_architecture.csv"
]

for filename in expected_files:

    path = OUTPUT_DIR / filename

    print(
        filename,
        ":",
        "CREATED" if path.exists() else "MISSING"
    )

print("\nAll Part 4 tasks completed.")
