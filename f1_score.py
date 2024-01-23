def f1_score(stats):
    """ Computes the f1 score

    Args:
        stats (dict): A dictionary containing the count of actual, predicted, and correct labels.

    Returns:
        float: The computed F1 score. Returns 0.0 if either the predicted count or the actual count is zero.
    """
    if stats['pred'] == 0 or stats['actual'] == 0:
        return 0.0

    precision = stats['corr'] / stats['pred']
    recall = stats['corr'] / stats['actual']

    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return f1 if stats['pred'] > 0 and stats['actual'] > 0 and f1 > 0 else 0.0
