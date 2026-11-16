from pathlib import Path
import subprocess
import sys

# Create the complete Part 3 Python script
script = r'''
import csv
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "records.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

with open(DATA_FILE, "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

feature_columns = [
    "months_registered",
    "missed_visit_count_12m",
    "copay_burden_ratio",
    "avg_bill_inr_thousands",
    "adherence_score"
]

X = np.array(
    [[float(row[col]) for col in feature_columns] for row in rows],
    dtype=float
)

y = np.array([int(row["no_show"]) for row in rows], dtype=int)
value = np.array(
    [float(row["care_plan_cost_inr_thousands"]) for row in rows],
    dtype=float
)

n = len(X)

# ============================================================
# TASK 3.1: STANDARDISATION AND K-MEANS FROM SCRATCH
# ============================================================

means = np.mean(X, axis=0)
stds = np.std(X, axis=0, ddof=1)

Z = (X - means) / stds

def kmeans(data, K):
    n_rows = len(data)

    # Pinned initial centroid positions
    step = n_rows // K
    centroids = data[
        [i * step for i in range(K)]
    ].copy()

    labels = np.full(n_rows, -1, dtype=int)
    assignment_passes = 0

    while True:
        # Squared Euclidean distances
        distances = np.sum(
            (data[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2
        )

        # argmin selects the lower centroid number in a tie
        new_labels = np.argmin(distances, axis=1)

        assignment_passes += 1

        changed = not np.array_equal(new_labels, labels)
        labels = new_labels

        if not changed:
            break

        # Update centroids
        for cluster in range(K):
            members = data[labels == cluster]

            if len(members) > 0:
                centroids[cluster] = np.mean(members, axis=0)

    # Calculate final WCSS
    wcss = 0.0

    for cluster in range(K):
        members = data[labels == cluster]

        if len(members) > 0:
            wcss += np.sum(
                (members - centroids[cluster]) ** 2
            )

    return labels, centroids, wcss, assignment_passes


results = {}
kmeans_results = {}

for K in range(2, 7):
    labels, centroids, wcss, passes = kmeans(Z, K)

    kmeans_results[K] = {
        "labels": labels,
        "centroids": centroids,
        "wcss": wcss,
        "passes": passes
    }

    results[f"wcss_k{K}"] = wcss

    print(
        f"K={K}, WCSS={wcss:.4f}, "
        f"assignment_passes={passes}"
    )

# ============================================================
# TASK 3.2: ELBOW METHOD
# ============================================================

elbow_k = None

for K in [3, 4, 5]:
    previous_drop = (
        kmeans_results[K - 1]["wcss"]
        - kmeans_results[K]["wcss"]
    )

    next_drop = (
        kmeans_results[K]["wcss"]
        - kmeans_results[K + 1]["wcss"]
    )

    if next_drop < 0.5 * previous_drop:
        elbow_k = K
        break

if elbow_k is None:
    raise RuntimeError("No elbow K found.")

results["elbow_k"] = elbow_k

print("\nTASK 3.2: ELBOW METHOD")
print("elbow_k:", elbow_k)

# ============================================================
# TASK 3.3: PROFILE THE K=3 SEGMENTS
# ============================================================

labels3 = kmeans_results[3]["labels"]

# Order clusters by ascending original X3 mean
cluster_order = sorted(
    range(3),
    key=lambda cluster: np.mean(
        X[labels3 == cluster, 2]
    )
)

segment_names = ["A", "B", "C"]
segment_indices = {}

segment_csv_rows = []

for name, cluster in zip(segment_names, cluster_order):
    indices = np.where(labels3 == cluster)[0]
    segment_indices[name] = indices

    segment_size = len(indices)

    segment_mean = np.mean(X[indices], axis=0)
    outcome_rate = np.mean(y[indices])
    mean_value = np.mean(value[indices])

    results[f"size_segment_{name}"] = segment_size

    segment_csv_rows.append([
        name,
        segment_size,
        *[f"{v:.4f}" for v in segment_mean],
        f"{outcome_rate:.4f}",
        f"{mean_value:.4f}"
    ])

    print(
        f"Segment {name}: size={segment_size}, "
        f"mean_X3={segment_mean[2]:.4f}, "
        f"outcome_rate={outcome_rate:.4f}"
    )

with open(
    OUTPUT_DIR / "part3_segments.csv",
    "w",
    newline=""
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "segment",
        "size",
        "mean_X1",
        "mean_X2",
        "mean_X3",
        "mean_X4",
        "mean_X5",
        "outcome_rate",
        "mean_value"
    ])

    writer.writerows(segment_csv_rows)

# ============================================================
# TASK 3.4: HIERARCHICAL CLUSTERING FROM SCRATCH
# ============================================================

cohort_names = ["E1", "E2", "E3", "E4", "E5", "E6"]

D = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])

def linkage_distance(cluster_a, cluster_b, method):
    distances = [
        D[i, j]
        for i in cluster_a
        for j in cluster_b
    ]

    if method == "single":
        return min(distances)

    if method == "complete":
        return max(distances)

    if method == "average":
        return sum(distances) / len(distances)

    raise ValueError("Unknown linkage method")


def cluster_name(cluster):
    return "+".join(
        cohort_names[i]
        for i in sorted(cluster)
    )


hierarchical_csv_rows = []

for method in ["single", "complete", "average"]:

    clusters = [[i] for i in range(6)]
    merge_heights = []

    for merge_no in range(1, 6):

        best_distance = float("inf")
        best_i = None
        best_j = None

        # Sort clusters by their smallest member
        clusters.sort(key=lambda c: min(c))

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                distance = linkage_distance(
                    clusters[i],
                    clusters[j],
                    method
                )

                if distance < best_distance:
                    best_distance = distance
                    best_i = i
                    best_j = j

        cluster_a = clusters[best_i]
        cluster_b = clusters[best_j]

        name_a = cluster_name(cluster_a)
        name_b = cluster_name(cluster_b)

        hierarchical_csv_rows.append([
            method,
            merge_no,
            name_a,
            name_b,
            f"{best_distance:.4f}"
        ])

        merge_heights.append(best_distance)

        merged_cluster = sorted(cluster_a + cluster_b)

        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in (best_i, best_j)
        ]

        clusters.append(merged_cluster)

    # Find the largest gap between consecutive merge heights
    gaps = [
        merge_heights[i + 1] - merge_heights[i]
        for i in range(4)
    ]

    largest_gap_index = int(np.argmax(gaps))

    merges_below_cut = largest_gap_index + 1
    clusters_at_cut = 6 - merges_below_cut

    results[f"{method}_clusters_at_cut"] = clusters_at_cut

    print(
        f"{method} linkage heights: {merge_heights}"
    )
    print(
        f"{method} clusters at cut: {clusters_at_cut}"
    )

with open(
    OUTPUT_DIR / "part3_hierarchical.csv",
    "w",
    newline=""
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "linkage",
        "merge_no",
        "cluster_a",
        "cluster_b",
        "height"
    ])

    writer.writerows(hierarchical_csv_rows)

# ============================================================
# TASK 3.5: PCA FROM SCRATCH
# ============================================================

# Step 1: Standardised data mean vector
pca_mean = np.mean(Z, axis=0)

# Step 2: Centre the data
Z_centered = Z - pca_mean

# Step 3: Covariance matrix
S = (Z_centered.T @ Z_centered) / (n - 1)

# Step 4: Eigen-decomposition
eigenvalues, eigenvectors = np.linalg.eigh(S)

# Step 5: Sort eigenvalues in descending order
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Step 6: Apply sign convention and keep top two components
for j in range(eigenvectors.shape[1]):
    largest_index = np.argmax(
        np.abs(eigenvectors[:, j])
    )

    if eigenvectors[largest_index, j] < 0:
        eigenvectors[:, j] *= -1

W = eigenvectors[:, :2]

# Step 7: Project data onto two principal components
projections = Z_centered @ W

# Eigenvalue results
eigenvalue_sum = np.sum(eigenvalues)

results["eigenvalue_sum"] = eigenvalue_sum

explained_ratios = eigenvalues / eigenvalue_sum
cumulative_ratios = np.cumsum(explained_ratios)

for i in range(5):
    results[f"eigenvalue_{i+1}"] = eigenvalues[i]
    results[f"explained_ratio_{i+1}"] = explained_ratios[i]
    results[f"cumulative_ratio_{i+1}"] = cumulative_ratios[i]

components_for_80pct = int(
    np.argmax(cumulative_ratios >= 0.80) + 1
)

results["components_for_80pct"] = components_for_80pct

# PC1 loadings
for i in range(5):
    results[f"pc1_loading_X{i+1}"] = W[i, 0]

# PC2 loadings
for i in range(5):
    results[f"pc2_loading_X{i+1}"] = W[i, 1]

# First three data-row projections
for row_index in range(3):
    results[f"row{row_index+1}_pc1"] = projections[row_index, 0]
    results[f"row{row_index+1}_pc2"] = projections[row_index, 1]

# Mean PC1 score for each segment
for name in ["A", "B", "C"]:
    indices = segment_indices[name]

    results[f"mean_pc1_segment_{name}"] = np.mean(
        projections[indices, 0]
    )

# ============================================================
# SAVE PART 3 RESULTS
# ============================================================

results_file = OUTPUT_DIR / "part3_results.txt"

with open(results_file, "w") as f:
    for key, val in results.items():

        if isinstance(val, (int, np.integer)):
            f.write(f"{key} = {int(val)}\n")

        else:
            f.write(f"{key} = {float(val):.4f}\n")

# ============================================================
# FINAL CHECKS
# ============================================================

print("\n======================================")
print("PART 3 RESULTS")
print("======================================")

for key, val in results.items():
    print(f"{key} = {val}")

print("\nCheckpoint WCSS K=2:")
print(f"{results['wcss_k2']:.4f}")

print("\nCheckpoint eigenvalue sum:")
print(f"{results['eigenvalue_sum']:.4f}")

print("\nTotal segment size:")
print(sum(results[f"size_segment_{s}"] for s in ["A", "B", "C"]))

print("\nTotal result keys:", len(results))

print("\nGenerated files:")

for filename in [
    "part3_results.txt",
    "part3_segments.csv",
    "part3_hierarchical.csv"
]:
    path = OUTPUT_DIR / filename
    print(filename, ":", path.exists())

print("\n======================================")
print("PART 3 COMPLETED")
print("======================================")
'''

# Save the script as part3.py
project_dir = Path("/content/Healthcare")
script_path = project_dir / "part3.py"

script_path.write_text(script, encoding="utf-8")

# Execute the script
subprocess.run(
    [sys.executable, str(script_path)],
    check=True
)

print("\nPart 3 script and all output files have been created.")
