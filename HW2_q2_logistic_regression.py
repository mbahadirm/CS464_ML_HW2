import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# === Load Data and Convert Labels ===
def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    label_map = {"low risk": 0, "mid risk": 1, "high risk": 1}
    y = df.iloc[:, -1].map(label_map).values
    return X, y

# === Sigmoid Function ===
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# === Predict Function ===
def predict(X, weights, bias):
    return (sigmoid(np.dot(X, weights) + bias) >= 0.5).astype(int)

# === Accuracy Function ===
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# === Confusion Matrix Function ===
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP],
                     [FN, TP]])

# === Training Function ===
def train_logistic_regression(X, y, val_X, val_y, lr, num_iter):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    val_acc_history = []

    for _ in range(num_iter):
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        error = y - y_pred
        dw = np.dot(X.T, error) / n_samples
        db = np.sum(error) / n_samples

        weights += lr * dw
        bias += lr * db

        val_preds = predict(val_X, weights, bias)
        val_acc = accuracy(val_y, val_preds)
        val_acc_history.append(val_acc)

    return weights, bias, val_acc_history

# === Main Function ===
def run_all():
    base_path = r"C:\Users\mbmut\OneDrive\Masaüstü\HW2\q2_q3_dataset"
    train_X, train_y = load_data(os.path.join(base_path, "train_data.csv"))
    val_X, val_y = load_data(os.path.join(base_path, "val_data.csv"))
    test_X, test_y = load_data(os.path.join(base_path, "test_data.csv"))

    # Normalize
    mean = np.mean(train_X, axis=0)
    std = np.std(train_X, axis=0)
    train_X = (train_X - mean) / std
    val_X = (val_X - mean) / std
    test_X = (test_X - mean) / std

    learning_rates = [1e-3, 1e-2, 1e-1, 1, 10]
    num_iter = 1000

    best_val_acc = 0
    best_test_acc = 0
    best_lr = None
    best_weights = None
    best_bias = None
    best_cm = None

    print("Learning Rate   Validation Acc  Test Acc")
    print("----------------------------------------")

    for lr in learning_rates:
        weights, bias, val_acc_history = train_logistic_regression(train_X, train_y, val_X, val_y, lr, num_iter)
        val_acc = val_acc_history[-1]

        test_preds = predict(test_X, weights, bias)
        test_acc = accuracy(test_y, test_preds)
        cm = confusion_matrix(test_y, test_preds)

        print(f"{lr:>7.4f}         {val_acc:.4f}          {test_acc:.4f}")
        print("Confusion Matrix:")
        print(cm)

        if (val_acc > best_val_acc) or (val_acc == best_val_acc and test_acc > best_test_acc):
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_lr = lr
            best_weights = weights
            best_bias = bias
            best_cm = cm

    print("\n===== FINAL REPORT =====")
    print(f"Best Learning Rate: {best_lr}")
    print(f"Test Accuracy: {best_test_acc:.4f}")
    print("Confusion Matrix:")
    print(best_cm)

if __name__ == "__main__":
    run_all()
