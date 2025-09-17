import torch
from transformers import Trainer
from typing import List
import numpy as np

def multi_label_class_weights(train_dataset) -> torch.Tensor:
    """
    Calculate inverse-frequency class weights for multi-label classification.
    Returns raw weights (no normalisation) suitable for BCEWithLogitsLoss(pos_weight=...).
    """
    labels = np.array([train_dataset[i]["labels"] for i in range(len(train_dataset))])
    class_counts = labels.sum(axis=0)

    # Avoid division by zero
    zero_mask = class_counts == 0
    if zero_mask.any():
        print(f"Warning: {zero_mask.sum()} classes have zero positives.")

    class_counts[zero_mask] = 1  # temporary fix to avoid division error

    total_samples = len(train_dataset)
    class_weights = total_samples / class_counts

    return torch.tensor(class_weights, dtype=torch.float)



class MultiClassTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=multi_class_class_weights(self.train_dataset, self.label_col)
        ).to(model.device)
        print(loss_fct.weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def multi_class_class_weights(train_dataset, label_col: str) -> torch.Tensor:
        # Convert to pandas for easier manipulation
        data = train_dataset.to_pandas()

        # Calculate the total number of samples
        total_samples = len(data)

        # Calculate the frequency of each class
        class_counts = data[label_col].value_counts()

        # Calculate weights for each class
        class_weights = []
        for count in class_counts:
            weight = total_samples / count
            class_weights.append(weight)

        # Convert to a tensor
        return torch.tensor(class_weights, dtype=torch.float32)