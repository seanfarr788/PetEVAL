from tqdm import tqdm
import numpy as np


def create_label_mapping(dataset):
    """
    Creates a mapping of unique labels to their corresponding BIO-tagged IDs.

    Args:
        dataset: A list of dictionaries containing input sentences and entities.

    Returns:
        label_mapping: A dictionary mapping BIO labels to their corresponding IDs.
    """
    unique_labels = set()

    for item in tqdm(dataset, desc="Extracting labels"):
        for entity in item.get("entities", []):
            label = entity.get("label")
            if label:  # Ensuring label is not None
                unique_labels.add(f"B-{label}")
                unique_labels.add(f"I-{label}")

    # Create label mapping, ensuring 'O' is mapped to 0
    label_mapping = {"O": 0}
    sorted_labels = sorted(unique_labels)

    # Assign IDs starting from 1
    label_mapping.update({label: idx for idx, label in enumerate(sorted_labels, start=1)})

    print(f'Available labels: {label_mapping}')
    return label_mapping



def tokenize_and_align_labels(
    examples, tokenizer, label_to_id, max_length, text_column="sentence", mode="train"
):
    """
    Tokenizes the input sentences and aligns the labels using BIO format.

    Args:
        examples: A dictionary containing input sentences and entity annotations.
        tokenizer: The tokenizer object.
        label_to_id: A dictionary mapping BIO labels to their corresponding IDs.
        max_length: Maximum length of tokenized input.
    """
    tokenized_inputs = tokenizer(
        examples[text_column],
        truncation=True,
        is_split_into_words=False,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,  # Return token index mappings
    )
    
    if mode == "infer":
        return tokenized_inputs
    
    labels = []
    for i, (sentence, entities) in enumerate(zip(examples[text_column], examples["entities"])):
        word_labels = ["O"] * len(sentence)  # Initialize all as "O"
        
        for entity in entities:
            start, end, label = entity["start"], entity["end"], entity["label"]
            word_labels[start] = f"B-{label}"  # Beginning of entity
            for j in range(start + 1, end):
                word_labels[j] = f"I-{label}"  # Inside entity
        
        # Align labels to tokenized words
        aligned_labels = []
        offset_mapping = tokenized_inputs["offset_mapping"][i]
        for offset in offset_mapping:
            if offset[0] == offset[1]:  # Special tokens
                aligned_labels.append(-100)
            else:
                token_start = offset[0]
                if token_start < len(word_labels):
                    aligned_labels.append(label_to_id.get(word_labels[token_start], 0))
                else:
                    aligned_labels.append(0)  # Default to "O"
        
        labels.append(aligned_labels)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs