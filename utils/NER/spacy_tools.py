from tqdm import tqdm


def prepare_spacy_dataset(dataset, text_col, label_col):
    """
    Function to prepare a Huggingface DatasetDict for SpaCy NER model training.
    Converts character-level entity spans to SpaCy-compatible format.

    Args:
        dataset (datasets.DatasetDict): A Huggingface DatasetDict containing 'savsnet_id',
            'sentence' and 'entities' columns with character-level entity spans.

    Returns:
        dict: A dictionary with train, dev (eval), and test splits, each containing
            lists of tuples formatted as (text, {"entities": [(start, end, label), ...]}).
    """
    spacy_dataset = {}

    for split in dataset.keys():
        spacy_data = []

        # Loop through each sentence and its entities in the dataset split
        for sentence_text, entities in tqdm(
            zip(dataset[split][text_col], dataset[split][label_col]),
            total=len(dataset[split][text_col]),
            desc=f"Processing {split} split...",
        ):
            entity_annotations = []

            # Sort entities by start position to avoid overlapping issues
            sorted_entities = sorted(entities, key=lambda x: x.get("start", 0))

            for entity in sorted_entities:
                char_start = entity.get("start")
                char_end = entity.get("end")
                label = entity.get("label")

                # Skip if any required values are missing
                if any(v is None for v in [char_start, char_end, label]):
                    tqdm.write(f"Skipping entity due to missing values: {entity}")
                    continue

                # Append entity in SpaCy format: (start, end, label)
                entity_annotations.append((char_start, char_end, label))

            # Add the sentence and its entities to the dataset in SpaCy format
            spacy_data.append((sentence_text, {"entities": entity_annotations}))

        spacy_dataset[split] = spacy_data

    return spacy_dataset
