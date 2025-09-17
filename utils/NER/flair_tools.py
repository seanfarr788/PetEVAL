from tqdm import tqdm
import torch
import logging
import pandas as pd
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

# Train Flair NER model
from flair.data import Sentence, Corpus, Span
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.datasets import FlairDatapointDataset
from flair.embeddings import TransformerWordEmbeddings


def prepare_flair_dataset(dataset, text_col: str, label_col: str) -> Corpus:
    """
    Function to prepare a Huggingface DatasetDict for FLAIR model training in NER.
    Converts character-level entity spans to word-level spans for Flair compatibility.

    Args:
        dataset (datasets.DatasetDict): A Huggingface DatasetDict containing 'savsnet_id',
            'sentence' and 'entities' columns with character-level entity spans.
    Returns:
        Corpus: A FLAIR Corpus object containing train, dev (eval), and test splits with
            tokenized sentences and word-level NER labels.
    """
    flair_dataset = {}

    for split in dataset.keys():
        sentences = []
        try:
            for sentence_text, entities in tqdm(
                zip(
                    dataset[split][text_col],
                    dataset[split][label_col],
                ),
                total=dataset[split].num_rows,
                desc=f"Processing {split} split...",
            ):
                # Create a FLAIR Sentence object
                sentence = Sentence(str(sentence_text))

                # Sort entities by start position to handle overlapping entities
                sorted_entities = sorted(entities, key=lambda x: x.get("start", 0))

                for entity in sorted_entities:
                    char_start = entity.get("start")
                    char_end = entity.get("end")
                    label = entity.get("label")

                    # Skip if any required values are missing
                    if any(v is None for v in [char_start, char_end, label]):
                        tqdm.write(f"Skipping entity due to missing values: {entity}")
                        continue

                    # Find the tokens that contain the start and end characters
                    start_token_idx = None
                    end_token_idx = None

                    for token_idx, token in enumerate(sentence.tokens):
                        token_start = token.start_position
                        token_end = token.end_position

                        # Find start token
                        if (
                            start_token_idx is None
                            and char_start >= token_start
                            and char_start <= token_end
                        ):
                            start_token_idx = token_idx

                        # Find end token
                        if char_end >= token_start and char_end <= token_end:
                            end_token_idx = token_idx
                            break

                    # Only create span if we found both start and end tokens
                    if start_token_idx is not None and end_token_idx is not None:
                        # Create span using token indices (inclusive)
                        span = Span(
                            tokens=sentence.tokens[start_token_idx : end_token_idx + 1]
                        )

                        # Add the label to the span
                        span.add_label("ner", label)

                    else:
                        tqdm.write(
                            f"Could not find matching tokens for entity: {entity}"
                        )
                        tqdm.write(f"In sentence: {sentence_text}")

                sentences.append(sentence)
        except Exception as e:
            for i in tqdm(
                range(0, len(dataset[split])),
                total=len(dataset[split]),
                desc=f"Processing {split} split...",
            ):
                sentence_text = str(dataset[split][i][text_col])
                sentences.append(Sentence(sentence_text))

        flair_dataset[split] = sentences
    try:
        # Create SentenceDataset for each split
        train_dataset = FlairDatapointDataset(flair_dataset["train"])
        eval_dataset = FlairDatapointDataset(flair_dataset["eval"])
        test_dataset = FlairDatapointDataset(flair_dataset["test"])
        # Create a Corpus using the train, eval, and test datasets
        corpus = Corpus(train=train_dataset, dev=eval_dataset, test=test_dataset)
    except:
        test_dataset = FlairDatapointDataset(flair_dataset["test"])
        corpus = Corpus(test=test_dataset)

    return corpus


def train_flair_ner_model(
    corpus: Corpus,
    pretrained_model: str = "bert-base-uncased",
    model_output_path: str = "/",
    learning_rate: float = 5e-5,
    mini_batch_size: int = 16,
    max_epochs: int = 50,
    hidden_size: int = 256,
    dropout: float = 0.1,
):
    """
    Enhanced function to train a FLAIR NER model with proper configuration and debugging.

    Args:
        corpus: A FLAIR Corpus object containing train, eval, and test datasets
        pretrained_model: Name of the pretrained transformer model
        model_output_path: Path where the trained model will be saved
        learning_rate: Learning rate for training
        mini_batch_size: Batch size for training
        max_epochs: Maximum number of training epochs
        hidden_size: Hidden size for the sequence tagger
        dropout: Dropout rate
        debug: Whether to print debug information

    Returns:
        SequenceTagger: The trained model if successful, None if there are issues
    """
    logging.info(
        f"[Training Parameters] Learning Rate: {learning_rate}, Batch Size: {mini_batch_size}, Max Epochs: {max_epochs}, Hidden Size: {hidden_size}, Dropout: {dropout}"
    )
    # Initialize embeddings with proper configuration
    embeddings = TransformerWordEmbeddings(
        model=pretrained_model,
        layers="all",  # Use last layer
        subtoken_pooling="first",
        fine_tune=False,
        use_context=True,
    )

    # Initialize sequence tagger
    tagger = SequenceTagger(
        hidden_size=hidden_size,
        embeddings=embeddings,
        tag_dictionary=corpus.make_label_dictionary(label_type="ner"),
        tag_type="ner",
        use_crf=True,
        use_rnn=True,
        rnn_layers=2,
        dropout=dropout,
        word_dropout=0.05,
        locked_dropout=0.5,
    )

    # Initialize trainer with proper configuration
    trainer = ModelTrainer(tagger, corpus)

    # Configure training
    train_parameters = {
        "learning_rate": learning_rate,
        "mini_batch_size": mini_batch_size,
        "mini_batch_chunk_size": 1,  # Adjust based on GPU memory
        "max_epochs": max_epochs,
        "patience": 2,  # Early stopping patience (if >2 epochs without improvement)
        "embeddings_storage_mode": "none",
        "optimizer": torch.optim.AdamW,
        "anneal_factor": 0.5,
        "min_learning_rate": 1e-7,
        "save_final_model": True,
        "shuffle": True,
    }
    trainer.train(
        base_path=model_output_path,
        **train_parameters,
    )
    return tagger


def test_set_eval(corpus: Corpus, model: SequenceTagger):
    """
    Function to evaluate a trained FLAIR NER model on the test set.

    Args:
        model: A trained FLAIR SequenceTagger model
        corpus: A FLAIR Corpus object containing train, eval, and test datasets

    Returns:
        dict: A dictionary containing the test set evaluation results
    """
    # Evaluate the model on the test set
    test_results = model.evaluate(
        corpus.test,
        gold_label_type="ner",
        mini_batch_size=32,
        out_path=f"predictions.txt",
    )
    print(f"\nTest Set Results: {test_results}")
    return test_results


def get_predictions(corpus, model):
    preds = []
    for sentence in tqdm(corpus.test, desc="Inferencing.."):
        model.predict(sentence)
        preds.append(sentence.to_dict(tag_type="ner"))
    return preds


def anonymisation(dataset, preds, text_col: str):
    """
    Anonymizes entities in text by replacing them with their respective tags.

    Args:
        dataset: List of dictionaries containing the text data
        predictions: List of dictionaries containing entity predictions
        text_column: Name of the column containing the text to anonymize

    Returns:
        List of dictionaries with anonymized text

    Example:
        predictions format:
        [{'entities': [{'text': 'John', 'labels': [{'value': 'PERSON'}],
                       'start': 0, 'end': 4}]}]
    """
    logging.info("Anonymizing entities...")

    # Create a deep copy to avoid modifying the original dataset
    processed_dataset = [{**item} for item in dataset]

    for idx in tqdm(range(len(dataset)), desc="Anonymizing..."):
        if idx >= len(preds):
            logging.warning(f"No predictions found for index {idx}")
            continue

        text = dataset[idx][text_col]
        if not text:
            logging.warning(f"Empty text found at index {idx}")
            continue

        # Sort entities by start position in reverse order
        # This ensures we process longer entities before shorter ones
        entities = sorted(
            preds[idx].get("entities", []),
            key=lambda x: (-x.get("start", 0), -len(x.get("text", ""))),
        )

        # Create a list of all positions that have been replaced
        replaced_positions = set()

        for entity in entities:
            try:
                start = entity.get("start_pos")
                end = entity.get("end_pos")
                target_text = entity.get("text")
                label = entity.get("labels", [{}])[0].get("value")

                if not all([start is not None, end is not None, target_text, label]):
                    logging.warning(f"Invalid entity at index {idx}: {entity}")
                    continue

                # Check if this position has already been replaced
                if any(pos in replaced_positions for pos in range(start, end)):
                    continue

                # Create the replacement tag
                replacement = f"<<{label}>>"

                # Replace the specific occurrence at the correct position
                text = text[:start] + replacement + text[end:]

                # Mark these positions as replaced
                replaced_positions.update(range(start, end))

            except Exception as e:
                logging.error(f"Error processing entity at index {idx}: {str(e)}")
                continue

        processed_dataset[idx][text_col] = text

    return processed_dataset


def save_predictions(preds, output_dir):
    logging.info(f"Saving predictions to {output_dir}")
    df = pd.DataFrame(preds)
    if output_dir.endswith(".csv"):
        df.to_csv(output_dir, index=False)
    else:
        df.to_csv(f"{output_dir}/predictions.csv", index=False)


def anonymise_dataframe(
    df: pd.DataFrame,
    text_column_name: str = "narrative",
    label_column_name: str = "entities",
) -> pd.DataFrame:
    current_directory = os.path.dirname(os.path.abspath(file))
    dataset = {"test": df.to_dict("records")}
    corpus = prepare_flair_dataset(
        dataset, text_col=text_column_name, label_col=label_column_name
    )
    flair_pretrained_model = "best-model.pt"
    logging.info(f"Inference with model from {flair_pretrained_model}")
    model = SequenceTagger.load(os.path.join(current_directory, flair_pretrained_model))
    # test_set_eval(corpus=corpus, model=model)
    preds = get_predictions(corpus=corpus, model=model)
    preds = anonymisation(dataset["test"], preds, text_column_name)
    return preds
