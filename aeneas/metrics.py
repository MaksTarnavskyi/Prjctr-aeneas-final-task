"""
Module with functions to calculate metrics during training evaluation
"""
from typing import List, Tuple, Dict
import numpy as np


def compute_metrics(p: Tuple[List[List[List[float]]], List[List[int]]]):
    """
    Function to produce custom metrics during training evaluation
    Args:
        p: tuple of predictions and labels

    Returns:
        Dict[str, float]: metric name to value mapping
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)  # predictions[:,:,1]

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [lab for (p, lab) in zip(prediction, label) if lab != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = evaluate(predictions=true_predictions, references=true_labels)
    return results


def evaluate(predictions: List[List[int]], references: List[List[int]]) -> Dict[str, float]:
    """Compute metrics on the validation set.

    Args:
        predictions:  predicted labels
        references:  true labels

    Returns:
        Dict[str, float]: metric name to value mapping

    """

    assert len(predictions) == len(references)

    tp = fp = tn = fn = 0

    for lineno, (preds, refs) in enumerate(zip(predictions, references), 1):

        if len(preds) != len(refs):
            msg = f"Error at line #{lineno}: number of tokens mismatch"
            raise ValueError(msg)

        tp += sum(1 for x, y in zip(refs, preds) if x == 1 and y >= 0.5)
        tn += sum(1 for x, y in zip(refs, preds) if x == 0 and y < 0.5)
        fp += sum(1 for x, y in zip(refs, preds) if x == 0 and y >= 0.5)
        fn += sum(1 for x, y in zip(refs, preds) if x == 1 and y < 0.5)

    try:
        prec = tp / (tp + fp)
    except ZeroDivisionError:
        prec = 1.0

    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return {
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "TN": tn,
        "Precision": prec,
        "Recall": recall,
        "F0.5": _f_measure(prec, recall, 0.5),
    }


def _f_measure(p, r, beta):
    """
    Args:
        p: Precision value
        r: Recall value
        beta: Beta value

    Returns:
        calculated F.beta measure
    """
    try:
        return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
    except ZeroDivisionError:
        return 0.0
