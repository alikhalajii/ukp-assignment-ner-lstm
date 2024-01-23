# file: sequence_evaluation.py

import numpy as np

def evaluate(predicted, actual, label_field, stats):
    """ Evaluate predictions and update stats.

    Args:
        predicted (array): Predicted label vector.
        actual (Tensor): Actual labels
        label_field (Field): Field of labels, containing the vocabulary
        stats (dict): Dictionary for returning the stats

    Returns:
        Dict: dictionary of spans
    """
    # Convert actual tensor to a numpy array
    actual_cpu = np.array(actual.t().cpu())
    actual_cpu = list(actual_cpu.reshape(-1))
        
    # Flatten the predicted labels
    pred_cpu = [l for sen in predicted for l in sen]

    # Map label IDs to their corresponding labels
    actual_spans = map_labels(actual_cpu, label_field.vocab.itos)
    pred_spans = map_labels(pred_cpu, label_field.vocab.itos)

    # Count the occurrences of actual and predicted labels
    count(actual_spans, pred_spans, stats)


def map_labels(ids, labels):
    """ Map ids of predictions to their labels

    Args:
        ids (list): IDs of predicted labels
        labels (list): List of labels in the vocab

    Returns:
        Dict: Dictionary of id to label
    """
    mapped = {}
    current = None
    lbl = None

    # Iterate through the predicted label IDs
    for i, id in enumerate(ids):
        l = labels[id]

        # Check if it's the beginning of a new span
        if l[0] == 'B':
            if current:
                mapped[lbl] = (current, i)

            current = l[2:]
            lbl = i

        elif l[0] == 'I':
            if current:
                if current != l[2:]:
                    mapped[lbl] = (current, i)
                    current = l[2:]
                    lbl = i
            else:
                # Start a new span if there was no previous span
                current = l[2:]
                lbl = i
        else:
            if current:
                mapped[lbl] = (current, i)
                current = None
                lbl = None
    return mapped

def count(actual, pred, stats):
    """Counts how often actual and predicted labels match and updates the stats dictionary.

    Args:
        actual (dict): A dictionary representing actual labels.
        pred (dict): A dictionary representing predicted labels.
        stats (dict): Resulting dictionary with stored count

    Returns:
        Dict: Dictionary of spans
    """

    # Update counts for actual labels
    for key, value in actual.items():
        if key not in stats:
            stats[key] = {'actual': 0, 'pred': 0, 'corr': 0}
        stats[key]['actual'] += 1

    # Update counts for predicted labels
    for key, value in pred.items():
        if key not in stats:
            stats[key] = {'actual': 0, 'pred': 0, 'corr': 0}
        stats[key]['pred'] += 1

    # Check if predicted labels match actual labels and update the count
    for key, (lbl, end) in actual.items():
        if key in pred:
            plbl, pend = pred[key]
            if lbl == plbl and end == pend:
                stats[key]['corr'] += 1