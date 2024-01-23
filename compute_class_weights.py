import torch
import numpy as np
from collections import Counter


def compute_class_weights(filename):
    """
    Compute class weights based on the frequency of each label in the dataset.

    Args:
        labels (list): List of labels in the dataset.

    Returns:
        torch.Tensor: Class weights as a tensor.
    """
    # Load your data and extract labels (replace this with your actual data loading logic)
    with open(filename, "r", encoding="utf-8") as file:
        lines = file.readlines()

    labels = [line.strip().split()[-1] for line in lines if line.strip()]

    # Map string labels to integer values
    label_to_index = {label: index for index, label in enumerate(set(labels))}
    indexed_labels = [label_to_index[label] for label in labels]

    # Count occurrences of each label in the entire dataset
    label_freqs = Counter(indexed_labels)

    # Compute class weights
    total_samples = len(indexed_labels)
    class_weights = torch.zeros(len(label_freqs))

    for label, freq in label_freqs.items():
        class_weights[label] = total_samples / (len(label_freqs) * freq)

    # Normalize class weights
    class_weights /= class_weights.sum()

    return class_weights

    """Debug
    # Print class weights
    print("\nClass Weights:", class_weights)
    print("Class Weights shaoe:", class_weights.shape)
    print("Class_weights Type", class_weights.dtype, "\n")

    # Use set to get unique labels
    unique_labels = set(labels)

    # Convert the set back to a list if needed
    unique_labels_list = list(unique_labels)

    # Print the unique labels
    print("\nUnique Labels:", unique_labels_list)
    num_classes = len(unique_labels)

    print("Number of classes:", num_classes, "\n")
    """





