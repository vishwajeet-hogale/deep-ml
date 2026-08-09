import numpy as np

def find_tpr_fpr(threshold, y_true, y_scores):
    # Convert the y_scores into 1 and 0 based on threshold
    y_scores, y_true = np.asarray(y_scores), np.asarray(y_true)
    y_pred = np.zeros(y_scores.shape)
    pos_mask = y_scores >= threshold
    y_pred[pos_mask] = 1

    # Calculate the confusion matrix 
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0)) 
    tn = np.sum((y_true == 0) & (y_pred == 0))

    return (tp/ (tp+fn), fp/(fp + tn))
    # return (TPR, FPR)
def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    thresholds = sorted(set(y_scores), reverse=True)
    thresholds = [np.inf] + thresholds
    tpr_vals, fpr_vals = [], []
    for threshold in thresholds:
        tpr, fpr = find_tpr_fpr(threshold, y_true, y_scores)
        
        tpr_vals.append(tpr)
        fpr_vals.append(fpr)

    return fpr_vals, tpr_vals





    