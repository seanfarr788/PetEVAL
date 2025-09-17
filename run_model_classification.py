# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")

parser = argparse.ArgumentParser(description="Description of your program")
parser.add_argument(
    "--problem_type",
    choices=["single_label_classification", "multi_label_classification"],
    help="Train a binary or multi-class model ('single_label_classification') or a multi-label model ('multi_label_classification')",
    default="single_label_classification",
)
parser.add_argument(
    "--mode",
    type=str,
    default="train",
    choices=["train", "eval", "infer"],
    help="Whether to train, evaluate or infer the model",
)

## Model ####
parser.add_argument(
    "--pretrained_model",
    help="Pretrained Model",
    default="SAVSNET/PetBERT",
)
parser.add_argument(
    "--tokenizer",
    help="Tokenizer. If none then the pretrained model tokenizer is used",
    default=None,
)
parser.add_argument(
    "--checkpoint",
    help="Path to the checkpoint",
    default=None,
)

## Data ####
parser.add_argument(
    "--dataset",
    help="Path to dataset. ",
    default="datasets/catergory/alopecia/arrow/",
)
parser.add_argument(
    "--text_column",
    default="text",
    help="text column")

parser.add_argument(
    "--label_column",
    default="label",
    help="label column")

## Training ####
parser.add_argument(
    "--epochs",
    default=None,
    help="Number of epochs. If none then early stopping is used",
)
parser.add_argument(
    "--patience",
    default=3,
    help="[if args.epochs == None] number of epochs to wait before early stopping",
),
parser.add_argument(
    "--batch_size",
    default=8,
    help="mini batch size for training")
parser.add_argument(
    "--threshold",
    default=None,
    help="threshold for each label. If none argmax is used (single label classification only)",
)

## Output ####
parser.add_argument(
    "--output_dir",
    default="datasets/catergory/alopecia/arrow/",
    help="output directory",
)
args = parser.parse_args()

"""

single_label_classification = When you want a model to produce only 1 possible label for each input. 
multi_label_classification = When you want a model to produce multiple possible labels for each input. 


# Example usage:
single_label_classification:
python run_model_classification.py --problem_type single_label_classification --mode train --pretrained_model "SAVSNET/PetBERT" --dataset "projects/PetEVAL/raw/train.csv" --text_column item_text --label_column label --output_dir "projects/PetEVAL/syndromic/train.csv"

multi_label_classification:
python run_model_classification.py --problem_type multi_label_classification --mode train --pretrained_model "SAVSNET/PetBERT" --dataset "projects/PetEVAL/raw/train.csv" --text_column item_text --label_column label --output_dir "projects/PetEVAL/syndromic/train.csv"


PetBERT-ICD inference:
python run_model_classification.py --problem_type multi_label_classification --mode infer --pretrained_model "SAVSNET/PetBERT_ICD" --dataset "projects/PetEVAL/raw/train.csv" --text_column item_text --output_dir "projects/PetEVAL/syndromic/train.csv"

"""

if __name__ == "__main__":
    logging.info(f"{args.mode}ing a {args.problem_type}...")
    if args.problem_type == "multi_label_classification":
        from scripts.classification.multi_label_trainer import main as multi_label
        multi_label(args)

    elif args.problem_type == "single_label_classification":
        from scripts.classification.multi_class_trainer import main as single_label
        single_label(args)
    else:
        raise ValueError(
            "problem_type must be either 'single_label_classification' or 'multi_label_classification'"
        )
