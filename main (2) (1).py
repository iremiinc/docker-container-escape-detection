import os
import re
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


DATASET_PATH = "Docker Container Escape(in).csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)

print("Original dataset shape:", df.shape)

df = df.drop_duplicates().reset_index(drop=True)

print("After duplicate removal:", df.shape)


def tokenize(value):
    if pd.isna(value):
        return []
    return re.findall(r"[A-Za-z0-9_./-]+", str(value))


for col in [
    "AddedCapabilitiesList",
    "DroppedCapabilitiesList",
    "SystemCallsList",
    "ReturnValueList"
]:
    df[col] = df[col].fillna("").astype(str)


df["Target"] = df["IsSafe"].astype(int)

added_tokens = df["AddedCapabilitiesList"].apply(tokenize)
dropped_tokens = df["DroppedCapabilitiesList"].apply(tokenize)
syscall_tokens = df["SystemCallsList"].apply(tokenize)
return_tokens = df["ReturnValueList"].apply(tokenize)


# =========================
# FEATURE ENGINEERING
# =========================

df["AddedCapabilityCount"] = added_tokens.apply(len)
df["DroppedCapabilityCount"] = dropped_tokens.apply(len)
df["NetCapabilityCount"] = df["AddedCapabilityCount"] - df["DroppedCapabilityCount"]
df["TotalCapabilityCount"] = df["AddedCapabilityCount"] + df["DroppedCapabilityCount"]

df["SystemCallCount"] = syscall_tokens.apply(len)
df["UniqueSystemCallCount"] = syscall_tokens.apply(lambda x: len(set(x)))
df["SystemCallDiversity"] = df["UniqueSystemCallCount"] / df["SystemCallCount"].replace(0, 1)

df["ReturnValueCount"] = return_tokens.apply(len)
df["UniqueReturnValueCount"] = return_tokens.apply(lambda x: len(set(x)))
df["ReturnValueDiversity"] = df["UniqueReturnValueCount"] / df["ReturnValueCount"].replace(0, 1)


capability_features = [
    "AddedCapabilityCount",
    "DroppedCapabilityCount",
    "NetCapabilityCount",
    "TotalCapabilityCount"
]

systemcall_features = [
    "SystemCallCount",
    "UniqueSystemCallCount",
    "SystemCallDiversity"
]

returnvalue_features = [
    "ReturnValueCount",
    "UniqueReturnValueCount",
    "ReturnValueDiversity"
]

all_features = capability_features + systemcall_features + returnvalue_features

X = df[all_features]
y = df["Target"]

print("\nClass distribution:")
print(y.value_counts())

print("\nUsed features:")
print(all_features)


# =========================
# CLASS DISTRIBUTION GRAPH
# =========================

plt.figure(figsize=(6, 4))
class_counts = y.value_counts().sort_index()
bars = plt.bar(["Vulnerable (0)", "Safe (1)"], class_counts.values)

for bar, value in zip(bars, class_counts.values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom",
        fontweight="bold"
    )

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/01_class_distribution.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# FEATURE MEANS BY CLASS GRAPH
# =========================

feature_means = df.groupby("Target")[all_features].mean().T

plt.figure(figsize=(12, 6))
x = np.arange(len(feature_means.index))
width = 0.35

plt.bar(x - width / 2, feature_means[0], width, label="Vulnerable (0)")
plt.bar(x + width / 2, feature_means[1], width, label="Safe (1)")

plt.title("Average Feature Values by Class")
plt.xlabel("Engineered Features")
plt.ylabel("Average Value")
plt.xticks(x, feature_means.index, rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/02_feature_means_by_class.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# =========================
# MODELS
# =========================

model_configs = {
    "Logistic Regression": {
        "features": capability_features,
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, C=0.001))
        ])
    },

    "K-Nearest Neighbors": {
        "features": systemcall_features,
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=301))
        ])
    },

    "Naive Bayes": {
        "features": returnvalue_features,
        "model": GaussianNB()
    },

    "SVM": {
        "features": returnvalue_features,
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="poly",
                degree=2,
                C=0.1,
                probability=True,
                random_state=42
            ))
        ])
    },

    "Random Forest": {
        "features": all_features,
        "model": RandomForestClassifier(
            n_estimators=50,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
    }
}


results = []
roc_data = []
confusion_matrices = {}


for model_name, config in model_configs.items():
    print("\nTraining:", model_name)

    selected_features = config["features"]
    model = config["model"]

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": pre,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": auc
    })

    print("Accuracy :", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall   :", round(rec, 4))
    print("F1-Score :", round(f1, 4))
    print("ROC-AUC  :", round(auc, 4))

    cm = confusion_matrix(y_test, y_pred)
    confusion_matrices[model_name] = cm

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Vulnerable", "Safe"]
    )

    disp.plot(values_format="d")
    plt.title("Confusion Matrix - " + model_name)
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_DIR}/confusion_matrix_{model_name.replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    roc_data.append((model_name, y_prob))


# =========================
# RESULTS TABLE
# =========================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="F1-Score", ascending=False)

print("\nFinal Model Comparison:")
print(results_df)

results_df.to_csv(f"{RESULTS_DIR}/model_comparison_results.csv", index=False)


# =========================
# ROC CURVE COMPARISON
# =========================

plt.figure(figsize=(9, 7))
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

for model_name, y_prob in roc_data:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_value = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.3f})")

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/03_roc_curve_comparison.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# BAR CHARTS FOR EACH METRIC
# =========================

for metric in ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]:
    plt.figure(figsize=(9, 5))
    bars = plt.bar(results_df["Model"], results_df[metric])

    for bar, value in zip(bars, results_df[metric]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    plt.title(metric + " Comparison")
    plt.xlabel("Models")
    plt.ylabel(metric)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(
        f"{RESULTS_DIR}/{metric.lower().replace('-', '_')}_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# =========================
# ALL METRICS GROUPED BAR CHART
# =========================

metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

plot_df = results_df.set_index("Model")[metrics]
plot_df.plot(kind="bar", figsize=(12, 6))

plt.title("Comprehensive Metrics Comparison Across ML Models")
plt.xlabel("Models")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=25, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/04_all_metrics_grouped_bar.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# MODEL PERFORMANCE HEATMAP
# =========================

heatmap_data = results_df.set_index("Model")[metrics]

plt.figure(figsize=(9, 4))
plt.imshow(heatmap_data.values, aspect="auto")
plt.colorbar(label="Score")

plt.xticks(np.arange(len(metrics)), metrics, rotation=25, ha="right")
plt.yticks(np.arange(len(heatmap_data.index)), heatmap_data.index)

for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        plt.text(
            j,
            i,
            f"{heatmap_data.values[i, j]:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold"
        )

plt.title("Model Performance Heatmap")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/05_model_performance_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# COMBINED CONFUSION MATRICES
# =========================

model_names = list(confusion_matrices.keys())

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, model_name in enumerate(model_names):
    cm = confusion_matrices[model_name]
    ax = axes[idx]

    ax.imshow(cm)
    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Vulnerable", "Safe"])
    ax.set_yticklabels(["Vulnerable", "Safe"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontweight="bold")

for idx in range(len(model_names), len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/06_combined_confusion_matrices.png", dpi=300, bbox_inches="tight")
plt.close()


# =========================
# PROJECT SUMMARY TXT
# =========================

best_model = results_df.iloc[0]

summary_text = f"""
DOCKER CONTAINER ESCAPE DETECTION - PROJECT SUMMARY

Dataset:
- Original dataset shape: {pd.read_csv(DATASET_PATH).shape}
- Dataset shape after duplicate removal: {df.shape}
- Target variable: IsSafe
- Positive class: Safe (1)
- Negative class: Vulnerable (0)

Preprocessing:
- Duplicate records were removed.
- ContainerName was excluded to reduce direct label leakage.
- Raw text columns were converted into numerical statistical features.
- Stratified train-test split was applied.

Feature Groups:
1. Capability Features:
{capability_features}

2. System Call Features:
{systemcall_features}

3. Return Value Features:
{returnvalue_features}

Models:
- Logistic Regression
- K-Nearest Neighbors
- Naive Bayes
- SVM
- Random Forest

Best Model:
- Model: {best_model['Model']}
- Accuracy: {best_model['Accuracy']:.4f}
- Precision: {best_model['Precision']:.4f}
- Recall: {best_model['Recall']:.4f}
- F1-Score: {best_model['F1-Score']:.4f}
- ROC-AUC: {best_model['ROC-AUC']:.4f}

Generated PNG Files:
- 01_class_distribution.png
- 02_feature_means_by_class.png
- 03_roc_curve_comparison.png
- 04_all_metrics_grouped_bar.png
- 05_model_performance_heatmap.png
- 06_combined_confusion_matrices.png
- Individual confusion matrices for each model
- Separate metric comparison charts
"""

with open(f"{RESULTS_DIR}/project_summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)


print("\nDone. All results and PNG files are saved in the 'results' folder.")
print("\nGenerated files:")
for file in sorted(os.listdir(RESULTS_DIR)):
    print(" -", file)