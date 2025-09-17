import torch
import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
)
from transformers import EvalPrediction
from typing import List
import logging
import pandas as pd


def multi_label_metrics(p: EvalPrediction, threshold=0.80):
    """
    Compute multi-label classification metrics.

    Args:
        predictions (array-like): The predicted outputs from the model, of shape (batch_size, num_labels).
        labels (array-like): The true labels, of shape (batch_size, num_labels).
        threshold (float, optional): The threshold value to convert probabilities to binary predictions. Default is 0.80.

    Returns:
        dict: A dictionary containing the following metrics:
            - 'f1': F1 score with micro average.
            - 'roc_auc': ROC AUC score with micro average.
            - 'accuracy': Accuracy score.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(preds))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    roc_auc = roc_auc_score(y_true, y_pred, average="micro")
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {"f1": f1_micro_average, "roc_auc": roc_auc, "accuracy": accuracy}
    return metrics


def multi_class_metrics(p):
    """
    Compute multi-class classification metrics.

    Args:
        p (tuple): A tuple containing the predictions and labels.

    Returns:
        dict: A dictionary containing the following metrics:
            - 'accuracy': Accuracy score.
            - 'precision': Precision score with micro average.
            - 'recall': Recall score with micro average.
            - 'f1': F1 score with micro average.
    """
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred, average="micro")
    precision = precision_score(y_true=labels, y_pred=pred, average="micro")
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def multi_label_evaluation(
    y_pred, y_true=None, labels=None, threshold: float = 0.50, report: bool = True
):
    """
    Passes predictions through sigmoid and applies threshold to convert probabilities to binary predictions.

    Args:
        y_pred (array-like): The predicted outputs from the model, of shape (batch_size, num_labels).
        y_true (array-like, optional): The true labels, of shape (batch_size, num_labels). Default is None.
        labels (list of str, optional): The list of label names. Default is None.
        threshold (float, optional): The threshold value to convert probabilities to binary predictions. Default is 0.80.
        report (bool, optional): Whether to print the classification report. Default is True.

    Returns:
        str: The classification report if report is True, otherwise None.
    """
    sigmoid = torch.nn.Sigmoid()
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.Tensor(y_pred)
    probs = sigmoid(y_pred)
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    if report:
        return classification_report(
            y_true, y_pred, target_names=labels, output_dict=True
        )
    else:
        return y_pred


def convert_predictions_to_labels(predictions, id_to_label):
    """
    Convert predictions to labels, skipping conversion if default Hugging Face labels are detected.

    Args:
        predictions (array-like): The predicted outputs from the model.
        id_to_label (list of str): The list of label names.

    Returns:
        list: The converted labels (or original indices if default labels are present).
    """
    if not isinstance(predictions, list):
        predictions = predictions.tolist()

    # Check if the id_to_label list contains the default Hugging Face labels
    if len(id_to_label) > 0 and all(
        isinstance(label, str) and label.startswith("LABEL_") for label in id_to_label
    ):
        return predictions  # Return original indices if default labels are found
    else:
        return [id_to_label[i] for i in predictions]


def multi_class_evaluation(
    y_pred: EvalPrediction,
    y_true: np.ndarray = None,
    labels: list[str] = None,
    threshold: float = None,
    report: bool = False,
):
    """
    Convert predictions to labels and optionally print a classification report.
    Handles both multi-class and multi-label scenarios based on the presence of a threshold.

    Args:
        y_pred (EvalPrediction): The predicted outputs from the model (logits).
        y_true (np.ndarray, list, optional): The true labels. Defaults to None.
        labels (list of str, optional): The list of label names. Defaults to None.
        threshold (float, optional): The threshold value (0 to 1) to convert probabilities
                                     to binary predictions for multi-label tasks.
                                     If None, assumes multi-class. Defaults to None.
        report (bool, optional): Whether to print and return the classification report.
                                 Defaults to False.

    Returns:
        Union[pd.DataFrame, np.ndarray, list, None]:
            - pd.DataFrame: The classification report if report is True.
            - np.ndarray or list: The converted predictions if report is False.
            - None: If y_pred.predictions is None.
    """
    if y_pred.predictions is None:
        return None

    if threshold is None:  # Multi-class scenario
        preds = np.argmax(y_pred.predictions, axis=-1)
        predictions = preds.flatten().tolist()
    else:  # Multi-label scenario with thresholding
        logging.info(f"Using threshold: {threshold}")
        probs = torch.sigmoid(torch.Tensor(y_pred.predictions)).numpy()
        predictions = (probs > threshold).astype(int).tolist()

    if report:
        if y_true is None:
            logging.warning("y_true is None, cannot generate classification report.")
            return None
        report_out = classification_report(
            y_true, predictions, target_names=labels, output_dict=True
        )
        logging.info(report_out)
        report_df = pd.DataFrame(report_out).transpose()
        return report_df
    else:
        if labels is not None:
            predictions = convert_predictions_to_labels(predictions, labels)
        return np.asarray(predictions) if threshold is None else predictions
