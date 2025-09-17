from datasets import Dataset, DatasetDict
import pandas as pd
import logging
from transformers import AutoTokenizer
from utils.training_tools import optimal_num_proc

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")


import logging
import pandas as pd
from datasets import Dataset

def import_dataset(train, eval=None, test=None):
    try:
        if train.endswith(".csv"):
            try:
                train_df = pd.read_csv(train)
            except Exception as e:
                raise RuntimeError(f"Failed to load training data from {train}: {e}")
            
            try:
                test_df = pd.read_csv(test)
            except Exception as e:
                raise RuntimeError(f"Failed to load test data from {test}: {e}")
            
            try:
                eval_df = pd.read_csv(eval)
            except Exception as e:
                logging.info(f"Failed to load eval data from {eval}: {e}. Taking 5% of training data as eval data.")
                eval_df = train_df.sample(frac=0.05)
                train_df = train_df.drop(eval_df.index)

        else:
            try:
                dataset = Dataset.load_from_disk(train)
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset from disk: {train}: {e}")
            
            try:
                train_df = dataset["train"].to_pandas()
            except Exception as e:
                raise RuntimeError(f"Failed to convert 'train' split to DataFrame: {e}")
            
            try:
                test_df = dataset["test"].to_pandas()
            except Exception as e:
                raise RuntimeError(f"Failed to convert 'test' split to DataFrame: {e}")
            
            try:
                eval_df = dataset["eval"].to_pandas()
            except Exception as e:
                logging.info(f"Failed to convert 'eval' split to DataFrame: {e}. Taking 5% of training data as eval data.")
                eval_df = train_df.sample(frac=0.05)
                train_df = train_df.drop(eval_df.index)

        logging.info(f"Train: {len(train_df)}, Eval: {len(eval_df)}, Test: {len(test_df)}")
        return train_df, eval_df, test_df

    except Exception as e:
        logging.error(f"Could not load dataset due to error: {e}")
        raise  # Optional: re-raise if you want the calling code to handle the error too



def create_arrow_dataset(train, eval, test):
    """
    Converts pandas DataFrames into Hugging Face Datasets and returns them as a DatasetDict.

    Args:
        train (pd.DataFrame): The training dataset in pandas DataFrame format.
        eval (pd.DataFrame): The evaluation dataset in pandas DataFrame format.
        test (pd.DataFrame): The test dataset in pandas DataFrame format.

    Returns:
        DatasetDict: A dictionary containing the Hugging Face Datasets with keys 'train', 'eval', and 'test'.
    """
    train_dataset = Dataset.from_pandas(train)
    eval_dataset = Dataset.from_pandas(eval)
    test_dataset = Dataset.from_pandas(test)
    huggingface_datasets = DatasetDict(
        {"train": train_dataset, "eval": eval_dataset, "test": test_dataset}
    )

    logging.info(f"Created Arrow dataset with keys: {huggingface_datasets.keys()}")
    return huggingface_datasets


def dataset_cleaner(df, text_column, label_column):
    """
    Cleans the dataset by renaming specified columns.

    Args:
        df (pd.DataFrame): The input dataframe to be cleaned.
        text_column (str): The name of the column to be renamed to 'text'.
        label_column (str or None): The name of the column to be renamed to 'label'.
                                    If None, no column will be renamed to 'label'.

    Returns:
        pd.DataFrame: The cleaned dataframe with renamed columns.
    """
    logging.info(f"Renaming columns to {text_column} to 'text'")
    df = df.rename(columns={text_column: "text"})
    if label_column is not None:
        try:
            df = df.rename(columns={label_column: "label"})
        except:
            logging.info(
                f"Column {label_column} not found in dataframes. Please check the column name and try again."
            )
    return df


def missing_rows(df, text_column, label_column):
    """
    Removes rows from the DataFrame that have missing values in the specified text and label columns.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process.
    text_column (str): The name of the text column to check for missing values.
    label_column (str): The name of the label column to check for missing values.

    Returns:
    pandas.DataFrame: The DataFrame with missing rows removed.
    """
    original_length = len(df)
    try:
        logging.info(f"Removing missing rows in {text_column} and {label_column}")
        df = df.dropna(subset=["text", "label"])
    except:
        logging.info(f"Removing missing rows in {text_column}")
        df = df.dropna(subset=["text"])
    logging.info(f"Number of rows: {len(df)} (Dropped: {original_length - len(df)})")
    return df.reset_index(drop=True)


def column_names(dataset, output):
    col_names = dataset["train"].column_names
    with open(f"{output}/columns.txt", "w") as f:
        for col in col_names:
            f.write(col + "\n")
            
def tokenizer_fn(tokenizer, examples, padding, truncation, max_length):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=8192)


def dataset_generator(args):
    logging.info(
        f"Loading data from {args.train_file}, {args.eval_file}, {args.test_file}"
    )
    train_df, eval_df, test_df = import_dataset(
        train=args.train_file, eval=args.eval_file, test=args.test_file
    )
    logging.info(f"Renaming columns to {args.text_column} to 'text'")
    train_df = dataset_cleaner(train_df, args.text_column, args.label_column)
    eval_df = dataset_cleaner(eval_df, args.text_column, args.label_column)
    test_df = dataset_cleaner(test_df, args.text_column, args.label_column)

    train_df = missing_rows(
        train_df, text_column=args.text_column, label_column=args.label_column
    )
    eval_df = missing_rows(
        eval_df, text_column=args.text_column, label_column=args.label_column
    )
    test_df = missing_rows(
        test_df, text_column=args.text_column, label_column=args.label_column
    )

    dataset = create_arrow_dataset(train=train_df, eval=eval_df, test=test_df)
    
    if args.tokenizer:
        logging.info(f"Tokenizing data using {args.tokenizer}")
        logging.info("Remember, tokenizers are model specific and therefore if you change to a different model, you may need to re-tokenize the data.")
        if args.padding == True:
            padding = "max_length"
        logging.info(f'Padding: {padding}, Truncation: {args.truncation}, Max Length: {args.max_length}')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        dataset = dataset.map(lambda examples: tokenizer(examples["text"], padding=padding, truncation=args.truncation, max_length=args.max_length), batched=True, num_proc=optimal_num_proc())

    logging.info(f"Saving data to {args.output_dir}")
    dataset.save_to_disk(args.output_dir)
    column_names(dataset, output=args.output_dir)
