import numpy as np
from itertools import combinations_with_replacement as cwr


def build_poly_design_matrix(X, degree):
    N, D = X.shape
    cols = [np.ones(N)]

    for d in range(1, degree + 1):
        for idx in cwr(range(D), d):
            cols.append(np.prod(X[:, idx], axis=1))
    return np.column_stack(cols)

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree,
                   bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_val, y_val = np.asarray(X_val), np.asarray(y_val)

    phi_val = build_poly_design_matrix(X_val, degree)

    train_error, val_error, used_sizes = [], [], []

    for n in train_sizes:
        n = min(int(n), X_train.shape[0])
        # print(n)
        if n < 1:
            continue

        X_n, y_n = X_train[:n], y_train[:n]          # rows, not columns
        phi_train = build_poly_design_matrix(X_n, degree)
        
        w = np.linalg.pinv(phi_train) @ y_n
        # print(w.shape)

        train_error.append(np.mean((y_n - phi_train @ w) ** 2))
        val_error.append(np.mean((y_val - phi_val @ w) ** 2))

    train_error = np.array(train_error)
    val_error = np.array(val_error)

    final_train = train_error[-1]
    gap = val_error[-1] - final_train

    high_bias = final_train > bias_threshold
    high_variance = gap > variance_threshold

    if high_bias:
        diagnosis = 'high_bias'
    elif high_variance:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'
        
    return {
        'train_errors' : train_error,
        'val_errors' : val_error,
        'diagnosis' : diagnosis
    }
    