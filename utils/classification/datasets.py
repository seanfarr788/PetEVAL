from datasets import (
    load_from_disk,
    load_dataset,
    Dataset,
    DatasetDict,
    Features,
    Value,
    ClassLabel,
)
import torch
from transformers import AutoTokenizer
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")


def multi_label_dataset(args):
    """
    Prepares a multi-label dataset for training or inference.

    Args:
        args: An object containing the following attributes:
            - pretrained_model (str): The name of the pretrained model to use for tokenization.
            - text_column (str): The name of the column containing the text data.
            - dataset (str): The path to the dataset.
            - do_train (bool): A flag indicating whether to prepare the dataset for training or inference.

    Returns:
        If `args.do_train` is True:
            tuple: A tuple containing:
                - ds_enc (Dataset): The tokenized and encoded dataset ready for training.
                - id2label (dict): A dictionary mapping label indices to label names.
                - label2id (dict): A dictionary mapping label names to label indices.
        If `args.do_train` is False:
            tuple: A tuple containing:
                - huggingface_datasets (DatasetDict): The tokenized and encoded dataset ready for inference.
                - None
                - None
    """

    def tokenize_and_encode(examples, tokenizer):
        return tokenizer(
            examples[args.text_column],
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
        )

    if args.dataset.endswith(".csv"):
        logging.info(f"Loading inference .CSV dataset from {args.dataset}")
        ds = pd.read_csv(args.dataset)
        ds_drop = ds.dropna(subset=[args.text_column])
        if len(ds) != len(ds_drop):
            logging.info(f"Dropping {len(ds) - len(ds_drop)} rows with missing text")

        test_dataset = Dataset.from_pandas(ds_drop)
        if args.tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer, problem_type="multi_label_classification"
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.pretrained_model, problem_type="multi_label_classification"
            )
        huggingface_datasets = DatasetDict({"test": test_dataset})
        huggingface_datasets = huggingface_datasets.map(
            lambda x: tokenizer(
                x[args.text_column],
                truncation=True,
                padding="max_length",
                max_length=tokenizer.model_max_length,
            ))
        return huggingface_datasets, None, None
    else:
        logging.info(f"Loading Arrow dataset from {args.dataset}")
        ds = load_from_disk(args.dataset)
        columns = ds["train"].column_names
        if args.mode in ['infer', 'eval']:
            if "train" in ds:
                del ds["train"]
            if "eval" in ds:
                del ds["eval"]
            
        # create labels column
        logging.info(f"Labeled columns: {columns}\n")
        for split in ds.keys():
            try:
                ds[split] = ds[split].remove_columns(['__index_level_0__'])
            except:
                pass
        ds = ds.map(
            lambda x: {
                "labels": torch.tensor([
                    x[c]
                    for c in columns
                    if c not in ["savsnet_consult_id", args.text_column]
                ], dtype=torch.float)
            }
        )
        logging.info(
            f'Only keeping "savsnet_consult_id", "{args.text_column}" and labels'
        )

        labels_id = [
            label
            for label in columns
            if label not in ["savsnet_consult_id", args.text_column, "labels"]
        ]
        id2label = {idx: label for idx, label in enumerate(labels_id)}
        label2id = {label: idx for idx, label in enumerate(labels_id)}

        if args.embedding_level == "word":
            logging.info(f"Tokenizing dataset using {args.pretrained_model}")
            if args.tokenizer:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.tokenizer, problem_type="multi_label_classification"
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.pretrained_model, problem_type="multi_label_classification"
                )
            logging.info(f"Max Sequence Length: {tokenizer.model_max_length}")
            ds_enc = ds.map(
                tokenize_and_encode,
                fn_kwargs={"tokenizer": tokenizer},
                batched=True,
                remove_columns=columns[0 : int(len(columns))],
            )
        elif args.embedding_level == "sentence":
            # SetFit does not require pre-tokenization
            ds_enc = ds
        else:
            raise ValueError(
                "embedding_level must be either 'word' or 'sentence' for multi-label classification"
            )

        ds_enc = ds_enc.map(
            lambda x: {"labels": torch.tensor(x["labels"], dtype=torch.float)},
            remove_columns=["labels"],
        )
        return ds_enc, id2label, label2id


def multi_class_dataset(dataset_path: str, text_col: str, tokenizer: AutoTokenizer, testing=False):
    """
    Preprocess a multi-class dataset for text classification using a tokenizer.

    Args:
        dataset_path (str): Path to the dataset. Should be a Huggingface dataset path.
        text_col (str): The name of the column containing the text data.

        tokenizer (PreTrainedTokenizer): A Huggingface tokenizer to preprocess the text data.

    Returns:
        Tuple[Dataset, PreTrainedTokenizer]: A tuple containing the tokenized dataset and the tokenizer.

    Raises:
        ValueError: If the dataset path points to a CSV file instead of a Huggingface dataset.
    """

    if ".csv" in dataset_path:
        logging.info("Loading dataset from a CSV file. Note that this can only be used for inference only")
        datasets = load_dataset("csv", data_files=dataset_path)
        datasets['test'] = datasets['train']
        # drop missing rows in text_col in datasets
        del datasets['train']
        
    else:
        datasets = load_from_disk(dataset_path)
        if datasets.keys() == None:
            logging.info("No split found in the dataset. Mapping to Test Set")
            datasets = datasets.train_test_split(test_size=1)
            datasets['test'] = datasets['train']
            del datasets['train']

    def preprocess_function(examples):
        return tokenizer(
            examples[text_col], padding=True, truncation=True, max_length=512
        )
    if testing == True:
        logging.info("Testing mode: Only using 100 samples")
        datasets = datasets.shuffle(seed=42).select(range(100))

    tokenized_dataset = datasets.map(
        preprocess_function,
        batched=True,
        num_proc=10,
        #remove_columns=[text_col],
    )
    return tokenized_dataset
