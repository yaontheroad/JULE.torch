import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def compute_nmi(labels_true, labels_pred):
    """
    Compute normalized mutual information between two clusterings.
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
    Returns:
        nmi: Normalized mutual information score
    """
    return normalized_mutual_info_score(labels_true, labels_pred)

def compute_ari(labels_true, labels_pred):
    """
    Compute adjusted Rand index between two clusterings.
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
    Returns:
        ari: Adjusted Rand index score
    """
    return adjusted_rand_score(labels_true, labels_pred)

def compute_acc(labels_true, labels_pred):
    """
    Compute clustering accuracy using Hungarian algorithm.
    Args:
        labels_true: Ground truth labels
        labels_pred: Predicted cluster labels
    Returns:
        acc: Clustering accuracy
    """
    from scipy.optimize import linear_sum_assignment
    
    labels_true = labels_true.astype(np.int64)
    labels_pred = labels_pred.astype(np.int64)
    
    # Get the unique labels
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    
    # Create confusion matrix
    confusion = np.zeros((len(unique_true), len(unique_pred)))
    for i in range(len(labels_true)):
        confusion[np.where(unique_true == labels_true[i])[0][0],
                 np.where(unique_pred == labels_pred[i])[0][0]] += 1
    
    # Find optimal one-to-one mapping
    row_ind, col_ind = linear_sum_assignment(-confusion)
    
    # Compute accuracy
    acc = confusion[row_ind, col_ind].sum() / len(labels_true)
    return acc 