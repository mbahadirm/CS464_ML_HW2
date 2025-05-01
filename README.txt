# CS464 HW2 – Machine Learning from Scratch

## Description

This project implements three foundational machine learning algorithms entirely **from scratch** without using external machine learning libraries:

- **Principal Component Analysis (PCA)**: A dimensionality reduction technique used to project high-dimensional cat and dog image data onto a lower-dimensional space. The program visualizes top eigenfaces and performs face morphing between species.

- **Logistic Regression**: A binary classification model trained with **batch gradient ascent** to predict maternal health risk (low vs. mid/high risk) based on medical features. Multiple learning rates are tested to optimize performance.

- **Support Vector Machine (SVM)**: A soft-margin linear SVM implemented using **hinge loss** and trained with batch gradient descent. A **5-fold cross-validation** is also implemented manually to tune the hyperparameter \( C \).

All models are evaluated using accuracy, confusion matrix, and other metrics (e.g., precision, recall, F1-score for SVM).

---

## Requirements

Make sure the following Python packages are installed:

- `numpy`
- `pandas`
- `matplotlib`
- `os`

Install them using:

```bash
pip install numpy pandas matplotlib
```

---

## Usage

To run each part of the homework, ensure the relevant `.py` files and dataset folders are in the same directory.

### Question 1 – PCA

```bash
python HW2_q1.py
python HW2_q1.3.py
```
- Outputs variance explained.
- Displays eigenfaces.
- Shows morphing from cat to dog.

### Question 2 – Logistic Regression

```bash
python HW2_q2_logistic_regression.py
```
- Tests multiple learning rates.
- Plots validation accuracy curves.
- Prints confusion matrices and selects best model based on **test accuracy**.

### Question 3 – Support Vector Machine (SVM)

```bash
python HW2_q3_svm.py
```
- Performs manual 5-fold cross-validation for each \( C \) value.
- Retrains with best \( C \) and evaluates on test set.
- Prints final test metrics (accuracy, precision, recall, F1-score).
---

## Notes

- Dataset files (train/val/test) should be located at the paths defined inside each script.
- All implementations follow mathematical formulations covered in class — without using machine learning toolkits like scikit-learn.
