from utils.dataset_tools.csv_2_huggingface import dataset_generator
import argparse

import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")

parser = argparse.ArgumentParser(
    description="Converts three pandas dataframes to Huggingface datasets and saves them to disk. Used for pretraining and classification pipelines."
)
###### Data Locations ######
parser.add_argument(
    "--train_file",
    default="train.csv",
)
parser.add_argument(
    "--eval_file",
    default="val.csv",
)
parser.add_argument(
    "--test_file",
    default="test.csv",
    help="test file",
)

###### Data Columns ######
parser.add_argument("--text_column", default="text", help="text column")
parser.add_argument(
    "--label_column",
    default="label",
    help="label column (For binary/multi-class classification)",
)

###### Output Directory ######
parser.add_argument(
    "--output_dir",
    default="out/",
    help="output directory",
)
args = parser.parse_args()

"""
Converts three pandas dataframes to Huggingface datasets and saves them to disk. Used for pretraining and classification pipelines.

Example usage:
python3 run_datasets_arrow_gen.py --train_file train.csv --eval_file eval.csv --test_file test.csv --text_column text --output_dir arrow
"""

if __name__ == "__main__":
    dataset_generator(args)
