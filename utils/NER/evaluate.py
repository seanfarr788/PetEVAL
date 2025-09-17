from seqeval.metrics import classification_report
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")



def compute_metrics(p, id2label):
    predictions, labels = p.predictions, p.label_ids
    predictions = np.argmax(predictions, axis=-1)

    # Convert integer labels to string labels
    true_labels = [[id2label[label] for label, pred in zip(label_row, pred_row) if label != -100]
                   for label_row, pred_row in zip(labels, predictions)]
    true_predictions = [[id2label[pred] for label, pred in zip(label_row, pred_row) if label != -100]
                        for label_row, pred_row in zip(labels, predictions)]
    
    # Flatten lists
    true_labels = [label for row in true_labels for label in row]
    true_predictions = [pred for row in true_predictions for pred in row]

    # Compute precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_predictions, average="macro")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logging.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return {"precision": precision, "recall": recall, "f1": f1}



def evaluate(tokenized_dataset, predictions, config, output_dir):
    """
    Evaluate NER model performance.

    Args:
        tokenized_dataset: Tokenized dataset containing input tokens and labels.
        predictions: Model predictions as token-level IDs.
        pad_token_id: ID of the padding token to exclude from evaluation.
        config: Model configuration containing label mappings.
        label_column: Column name containing true labels.
        output_dir: Directory to save evaluation results.
    """
    # Get predictions and true labels
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = tokenized_dataset["test"]["labels"]

    # Convert token IDs to labels
    pred_labels = [
        [config.id2label[label] for label in example if label != -100]
        for example in pred_ids
    ]

    true_labels = [
        [config.id2label[label] for label in example if label != -100]
        for example in true_ids
    ]

    report = classification_report(true_labels, pred_labels, output_dict=True)
    report = pd.DataFrame(report).transpose()

    print(report)

    if output_dir.endswith(".csv"):
        report.to_csv(output_dir, index=True)
    else:
        report.to_csv(f"{output_dir}/transformers_results.csv", index=True)


def extract_entities(dataset, predictions, tokenizer, id2label):
    """
    Extract named entities from model predictions and add them to the dataset efficiently.
    
    Args:
        dataset: The HuggingFace dataset (test split)
        predictions: Output from trainer.predict(dataset['test'])
        tokenizer: The tokenizer used for the model
        id2label: Mapping from prediction IDs to entity labels
        
    Returns:
        Dataset with a new 'entities' column containing extracted entities
    """
    pred_label_ids = np.argmax(predictions.predictions, axis=-1)  # Vectorized argmax
    
    all_entities = []
    
    special_tokens = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
    
    for i, example in tqdm(enumerate(dataset), total=len(dataset), desc="Extracting entities"):
        input_ids = example["input_ids"]
        pred_ids = pred_label_ids[i]
        
        # Decode tokens & filter special tokens in one pass
        tokens, labels = zip(*[
            (token, pred_id)
            for token, pred_id in zip(tokenizer.convert_ids_to_tokens(input_ids), pred_ids)
            if token not in special_tokens and pred_id != -100
        ])
        
        # Extract entities using BIO scheme
        entities = []
        current_entity = None
        current_tokens = []

        for token, pred_id in zip(tokens, labels):
            label = id2label[pred_id]

            if label.startswith("B-"):
                # Save previous entity
                if current_entity:
                    entities.append({"label": current_entity, "text": tokenizer.convert_tokens_to_string(current_tokens).strip()})
                
                # Start a new entity
                current_entity = label[2:]  # Remove "B-" prefix
                current_tokens = [token]

            elif label.startswith("I-") and current_entity == label[2:]:
                current_tokens.append(token)
            
            else:
                # If switching to "O" or different entity, finalize the last entity
                if current_entity:
                    entities.append({"label": current_entity, "text": tokenizer.convert_tokens_to_string(current_tokens).strip()})
                    current_entity = None
                    current_tokens = []

        # Save the last entity if it exists
        if current_entity:
            entities.append({"label": current_entity, "text": tokenizer.convert_tokens_to_string(current_tokens).strip()})

        all_entities.append(entities)

    # More memory-efficient dataset update
    return dataset.add_column("new_entities", all_entities)