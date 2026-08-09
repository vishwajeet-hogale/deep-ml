import numpy as np

def confusion_matrix(y_true, y_scores, threshold):
    y_pred = np.zeros(y_scores.shape[0])
    pred_mask = y_scores >= threshold
    y_pred[pred_mask] = 1


    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if int(np.sum((y_pred == 0))) == y_scores.shape[0]:
        return 1.0, tp / (tp + fn)
    
    elif int(np.sum((y_true == 0))) == y_scores.shape[0]:
        return tp / (tp + fp), 0.0

    return tp / (tp + fp), tp / (tp + fn)

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    # Find the thresholds in unique descending order
    thresholds = sorted(set(y_scores), reverse = True)

    # For each threshold you need to find the tp, fp, tn, fn 
    y_true, y_scores = np.asarray(y_true), np.asarray(y_scores)
    prec, rec, thres = [], [], []
    for threshold in thresholds:
        # Store the precision, recall and threshold in seperate lists 
        p, r = confusion_matrix(y_true, y_scores, threshold)
        prec.append(float(p))
        rec.append(float(r))
        thres.append(threshold)

    return (prec, rec, thres)

    # Return a tuple (prec, rec, threshold)