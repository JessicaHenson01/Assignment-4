"""
Module: test.py

This module provides functions for evaluating a video classification model on test data.
It includes functions to compute predictions and accuracy, generate a detailed classification report,
and compute a multilabel confusion matrix for all classes.

Functions:
    - test: Evaluates the model on a test DataLoader and returns the ground truth labels,
      predicted labels, and overall accuracy.
    - get_test_report: Generates a classification report using scikit-learn's classification_report.
    - get_confusion_matrix: Computes a multilabel confusion matrix for each class.
"""

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, multilabel_confusion_matrix

def test(model, dataloader, device, criterion):
    """
    Evaluate the model on the test dataset and compute overall accuracy.
    
    This function sets the model to evaluation mode and processes the test data
    from the provided DataLoader. It computes predictions for each batch, counts the number
    of correct predictions, and accumulates the true and predicted labels.
    
    Args:
        model (torch.nn.Module): The trained video classification model.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the test dataset.
        device (torch.device): The device (CPU or GPU) on which to perform evaluation.
    
    Returns:
        tuple: (targets, outputs, accuracy)
            - targets (list): Ground truth labels for all samples.
            - outputs (list): Predicted labels for all samples.
            - accuracy (float): Overall accuracy computed as the ratio of correct predictions
                                to the total number of samples.
    """
    model.eval()
    total_correct_preds = 0
    running_loss = 0.0
    len_dataset = len(dataloader.dataset)
    targets, outputs = [], []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Multi-clip input:
            # (batch, views, frames, channels, height, width)
            if x_batch.ndim == 6:
                (
                    batch_size,
                    number_of_views,
                    frame_count,
                    channels,
                    height,
                    width,
                ) = x_batch.shape

                flattened_clips = x_batch.reshape(
                    batch_size * number_of_views,
                    frame_count,
                    channels,
                    height,
                    width,
                )

                view_logits = model(flattened_clips)

                view_logits = view_logits.reshape(
                    batch_size,
                    number_of_views,
                    -1,
                )

                # Average the predictions from beginning, middle, and end.
                logits = view_logits.mean(dim=1)

            else:
                # Keep supporting ordinary single-clip evaluation.
                logits = model(x_batch)

            loss = criterion(logits, y_batch)
            running_loss += loss.item() * x_batch.size(0)

            predictions = logits.argmax(dim=1)
            total_correct_preds += (
                predictions == y_batch
            ).sum().item()

            outputs.extend(
                predictions.detach().cpu().numpy().tolist()
            )
            targets.extend(
                y_batch.detach().cpu().numpy().tolist()
            )

    test_loss = running_loss / len_dataset
    accuracy = total_correct_preds / len_dataset

    return targets, outputs, test_loss, accuracy

def get_test_report(target, output, target_names):
    """
    Generate a detailed classification report based on test results.
    
    This function uses scikit-learn's classification_report to produce a dictionary
    containing precision, recall, F1-score, and support for each class.
    
    Args:
        target (list): Ground truth labels.
        output (list): Predicted labels.
        target_names (list): List of class names corresponding to the labels.
    
    Returns:
        dict: A classification report as a dictionary.
    """
    return classification_report(target, output, output_dict=True, target_names=target_names)

def get_confusion_matrix(targets, outputs, labels_dict, all_cats):
    """
    Compute a multilabel confusion matrix for each class.
    
    This function converts numeric labels to their corresponding class names using the provided
    labels_dict, then computes a multilabel confusion matrix for each class using scikit-learn.
    
    Args:
        targets (list): Ground truth numeric labels.
        outputs (list): Predicted numeric labels.
        labels_dict (dict): Dictionary mapping class names to numeric labels.
        all_cats (list): List of all class names.
    
    Returns:
        dict: A dictionary where keys are class names and values are the corresponding confusion matrices.
    """
    # Create an inverse mapping from numeric label to class name
    inv_labels_dict = {label: cat for cat, label in labels_dict.items()}
    target_cats = [inv_labels_dict[target] for target in targets]
    output_cats = [inv_labels_dict[output] for output in outputs]
    confusion_mat = multilabel_confusion_matrix(target_cats, output_cats, labels=all_cats)
    return {label: mat for label, mat in zip(all_cats, confusion_mat)}
