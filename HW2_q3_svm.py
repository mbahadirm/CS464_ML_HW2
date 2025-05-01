import numpy as np
import pandas as pd
import os
from sklearn.svm import SVC

# === Load Data and Convert Labels ===
def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    label_map = {"low risk": 0, "mid risk": 1, "high risk": 1}
    y = df.iloc[:, -1].map(label_map).values
    return X, y

# === Predict and Metrics ===
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TP, FP], [FN, TN]])

def classification_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP, FP = cm[0]
    FN, TN = cm[1]
    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return accuracy, precision, recall, f1, cm

# === K-Fold Split Function ===
def k_fold_split(X, y, k):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = len(X) if i == k - 1 else (i + 1) * fold_size
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])
        folds.append((X[train_idx], y[train_idx], X[val_idx], y[val_idx]))
    return folds

# === Main Execution ===
def run_all():
    base_path = r"C:\Users\mbmut\OneDrive\Masaüstü\HW2\q2_q3_dataset"
    train_X, train_y = load_data(os.path.join(base_path, "train_data.csv"))
    test_X, test_y = load_data(os.path.join(base_path, "test_data.csv"))

    # Normalize
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0)
    train_X = (train_X - mean) / std
    test_X = (test_X - mean) / std

    Cs = [0.001, 0.01, 0.1, 1, 10]
    print("===== 5-Fold Cross-Validation Results =====")
    print(f"{'C':>8} | {'Mean Acc':>9} | {'Std Dev':>8}")
    print("-" * 32)

    best_C = None
    best_acc = 0
    cv_results = {}

    for C in Cs:
        folds = k_fold_split(train_X, train_y, k=5)
        accs = []
        for fold_X, fold_y, val_X, val_y in folds:
            model = SVC(kernel='linear', C=C)
            model.fit(fold_X, fold_y)
            preds = model.predict(val_X)
            acc, _, _, _, _ = classification_metrics(val_y, preds)
            accs.append(acc)
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        cv_results[C] = (mean_acc, std_acc)
        print(f"{C:8.3f} | {mean_acc:9.4f} | {std_acc:8.4f}")
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_C = C

    print("\n===== TRAINING ON FULL TRAIN SET =====")
    final_model = SVC(kernel='linear', C=best_C)
    final_model.fit(train_X, train_y)
    final_preds = final_model.predict(test_X)

    acc, prec, rec, f1, cm = classification_metrics(test_y, final_preds)

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Selected C: {best_C}")
    print(f"Test Accuracy: {acc:.4f}")
    print("Confusion Matrix (Prediction vs Actual):")
    print(cm)
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

if __name__ == "__main__":
    run_all()
