import spacy
from tqdm import tqdm
from spacy.training import Example
import logging
import random
from spacy.util import compounding, minibatch
from utils.dataset_tools.load import load_dataset_file
from utils.NER.spacy_tools import (
    prepare_spacy_dataset,
)
import os

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
from utils.training_tools import epoch_strategy


def get_model(model_path):
    """
    Load a spaCy model from the given path or create a blank English model if no path is provided.

    Args:
        model_path (str): The path to the spaCy model to load. If None, a blank English model is created.

    Returns:
        spacy.language.Language: The loaded or created spaCy model.

    Raises:
        ValueError: If the model cannot be loaded from the provided path.

    Notes:
        - If the model path is provided and the model cannot be loaded, an error is logged and a ValueError is raised.
        - If the model path is None, a blank English model is created and a corresponding info message is logged.
        - The function ensures that the 'ner' pipeline component is present in the model.
    """
    if model_path is not None:
        try:
            model = spacy.load(model_path)
            logging.info("Loaded model '%s'", model_path)
        except OSError:
            logging.error(
                f'Could not load model "{model_path}. You may need to run "python -m spacy download {model_path}" first"'
            )
            raise ValueError(
                f"Could not load model {model_path}. Ensure the model path is correct and the model is properly installed."
            )
    else:
        model = spacy.blank("en")
        logging.info("Created blank 'en' model")

    # Set up the pipeline
    if "ner" not in model.pipe_names:
        ner = model.add_pipe("ner")
        logging.info("Added 'ner' pipeline component to the model")
    else:
        ner = model.get_pipe("ner")
        logging.info("Found 'ner' pipeline component in the model")

    return model


def train(args):
    """
    Train a Named Entity Recognition (NER) model using spaCy.

    Args:
        args: An object containing the following attributes:
            - dataset (str): Path to the dataset file.
            - text_column (str): Name of the column containing text data.
            - entity_column (str): Name of the column containing label data.
            - pretrained_model (str): Name or path of the pretrained spaCy model to use.
            - epochs (int): Number of training epochs.
            - output_dir (str): Directory to save the trained model.

    Returns:
        None
    """
    # Load dataset
    dataset = load_dataset_file(args.dataset)
    corpus = prepare_spacy_dataset(
        dataset, text_col=args.text_column, label_col=args.entity_column
    )

    model = get_model(args.pretrained_model)

    # Add labels to the NER pipeline
    for _, annotations in corpus["train"]:
        for ent in annotations.get("entities"):
            model.add_label(ent[2])

    args, callbacks = epoch_stratergy(args)

    eval_loss = []
    best_score = 0
    no_improve_epochs = 0

    for epoch in tqdm(range(args.epochs)):
        print(f"Epoch {epoch+1}/{args.epochs}")
        losses = {}

        examples = corpus["train"]
        random.shuffle(examples)

        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))

        for batch in tqdm(batches, desc="Training", leave=False):
            texts, annotations = zip(*batch)
            example_objs = [
                Example.from_dict(model.make_doc(text), annot)
                for text, annot in zip(texts, annotations)
            ]
            model.update(example_objs, drop=0.5, losses=losses)

        tqdm.write(f"Losses: {losses}")

        report = evaluate(model, corpus)  # Evaluate the model
        eval_score = report["ents_f"]
        eval_loss.append(eval_score)

        # check to see if list is empty
        if callbacks:
            if eval_score > best_score:
                best_score = eval_score
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= args.patience:
                print("Early stopping triggered due to no improvement.")
                break

    # Save the model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model.to_disk(args.output_dir)
        print(f"Model saved to {args.output_dir}")


def inference(args):
    """
    Perform Named Entity Recognition (NER) inference using a pre-trained spaCy model on a given dataset.

    Args:
        args: An object containing the following attributes:
            pretrained_model (str): The name or path of the pre-trained spaCy model to use.
            dataset (str): The path to the dataset file to perform inference on.
            text_column (str): The name of the column in the dataset that contains the text to analyze.
            output_dir (str): The directory where the output CSV file with the inferred entities will be saved.

    Returns:
        None: The function saves the results to a CSV file in the specified output directory.
    """
    model = get_model(args.pretrained_model)  # python -m spacy download en_core_web_sm
    dataset = load_dataset_file(args.dataset)
    entities = []
    for example in tqdm(dataset["test"]):
        doc = model(example[args.text_column])
        example["entities"] = [(ent.text, ent.label_) for ent in doc.ents]
        entities.append(example["entities"])

    dataset = dataset["test"].to_pandas()
    dataset["entities"] = entities
    dataset.to_csv(f"{args.output_dir}/spacy_inference.csv", index=False)


def evaluate(model, corpus, split="test"):
    """
    Evaluate the given model on the test set from the provided corpus.

    Args:
        model (spacy.language.Language): The spaCy model to be evaluated.
        corpus (dict): A dictionary containing the test set with texts and their annotations.

    Returns:
        None

    Prints:
        Precision, Recall, F1-score, and Losses of the model on the test set.
    """
    if split == "eval" and "eval" not in corpus:
        logging.error("No evaluation set found in the corpus.")
        return None
    if split == "eval" and "eval" in corpus:
        examples = [
            Example.from_dict(model.make_doc(text), annot)
            for text, annot in corpus["eval"]
        ]
        results = model.evaluate(examples)
        tqdm.write("Evaluation set evaluation:")
        tqdm.write(f"Precision: {results['ents_p']:.3f}")
        tqdm.write(f"Recall: {results['ents_r']:.3f}")
        tqdm.write(f"F1-score: {results['ents_f']:.3f}")
        tqdm.write(f"Losses: {results['losses']}")
        return results
    # Evaluate the model on the test set
    if split == "test" and "test" not in corpus:
        print("No test set found in the corpus.")
        return None
    if split == "test" and "test" in corpus:
        examples = [
            Example.from_dict(model.make_doc(text), annot)
            for text, annot in corpus["test"]
        ]
        results = model.evaluate(examples)
        print("Test set evaluation:")
        print(f"Precision: {results['ents_p']:.3f}")
        print(f"Recall: {results['ents_r']:.3f}")
        print(f"F1-score: {results['ents_f']:.3f}")
        print(f"Losses: {results['losses']}")


def main(args):
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        inference(args)
    elif args.mode == "eval":
        nlp = get_model(args.pretrained_model)
        dataset = load_dataset_file(args.dataset)
        evaluate(nlp, dataset["test"])
    else:
        raise ValueError("Invalid mode. Choose either 'train' or 'infer'.")
