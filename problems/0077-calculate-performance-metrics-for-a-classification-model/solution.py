from collections import Counter

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    data = list(zip(actual, predicted))
    data_counts = Counter(data)

    # Confusion matrix components
    TP = data_counts[(1, 1)]
    FN = data_counts[(1, 0)]
    FP = data_counts[(0, 1)]
    TN = data_counts[(0, 0)]

    confusion_matrix = (
        (TP, FN),
        (FP, TN)
    )

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total if total else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    specificity = TN / (FP + TN) if (FP + TN) else 0.0
    negative_predictive = TN / (TN + FN) if (TN + FN) else 0.0

    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) else 0.0

    return (
        confusion_matrix,
        round(accuracy, 3),
        round(f1, 3),
        round(specificity, 3),
        round(negative_predictive, 3),
    )
