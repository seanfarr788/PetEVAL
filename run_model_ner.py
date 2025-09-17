# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from scripts.NER.ner_transformer import main as transformer_main
import argparse

import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    help="Backend for the model options",
    default="transformers",
)
parser.add_argument(
    "--mode",
    type=str,
    default="eval",
    choices=["train", "eval", "infer", "inference"],
    help="Whether to train, evaluate or infer the model",
)
parser.add_argument(
    "--pretrained_model",
    type=str,
    default="SAVSNET/PetHarbor",
    help="Name of the pretrained transformer model, used for fine-tuning",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to the checkpoint file for the (transformer) model",
)
parser.add_argument(
    "--tokenizer",
    type=str,
    default="SAVSNET/PetBERT",
    help="Name of the tokenizer to use for the transformer model. If None, pretrained_model path used.",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="SAVSNET/PetEVAL",
    help="Path to the dataset. For training a Huggingface dataset is required. For inference, either csv or the test set of Huggingface dataset",
)
parser.add_argument(
    "--text_column",
    type=str,
    default="sentence",
    help="Name of the column containing text in the dataset",
)
parser.add_argument(
    "--entity_column",
    type=str,
    default="annonymisation",
    help="Name of the column containing NER tags in the dataset",
)
parser.add_argument(
    "--anonymisation",
    type=bool,
    default=False,
    help="If true, the entities are replaced with their respective tags",
)
parser.add_argument(
    "--context_length",
    type=int,
    default=512,
    help="Maximum length of the input text",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=None,
    help="number of epochs. If none then early stopping is used",
)
parser.add_argument(
    "--batch_size", type=int, default=32, help="Batch size for training"
)
parser.add_argument(
    "--learning_rate", type=float, default=5e-5, help="Learning rate for training"
)
parser.add_argument(
    "--hidden_size", type=int, default=256, help="Hidden size for the sequence tagger"
)
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
parser.add_argument(
    "--patience", type=int, default=3, help="Patience for early stopping"
)
parser.add_argument(
    "--device", type=str, default="cuda:0", help="Device to use (cpu or cuda)"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="out",
    help="Output directory",
)
args = parser.parse_args()


"""
Script for training a Named Entity Recognition (NER) model.
This script supports training from scratch or fine-tuning a pretrained model 
using a Huggingface dataset. It also includes options for evaluation, anonymization, 
and hyperparameter configuration.

Transformers Library: https://huggingface.co/transformers/

Arguments:
    --pretrained_model (str): Name of the pretrained transformer model to use for 
        fine-tuning. Defaults to "SAVSNET/PetBERT".
    --dataset (str): Path to the dataset for training or inference. During training, 
        the dataset must be a Huggingface dataset. For inference, it can either be a 
        CSV file or the test set of the Huggingface dataset.
    --text_column (str): Name of the column in the dataset that contains the input 
        text. Defaults to "sentence".
    --label_column (str): Name of the column in the dataset that contains the NER tags. 
        Defaults to "entities".
    --anonymisation (bool): If set to True, replaces entities in the dataset with their 
        corresponding tags for anonymization.
    --max_epochs (int): Maximum number of training epochs. Early stopping is applied to 
        terminate training if no improvement is detected. Defaults to 500.
    --batch_size (int): Batch size for training. Defaults to 64.
    --learning_rate (float): Learning rate for the optimizer. Defaults to 5e-5.
    --hidden_size (float): Size of the hidden layers in the sequence tagger model. 
        Defaults to 256.
    --dropout (float): Dropout rate for regularization during training. Defaults to 0.1.
    --output_dir (str): Directory where the trained model, evaluation results, and other 
        outputs will be saved.

Example Transformers usage:
-- Training Transformers Model:
python run_model_ner.py --backend "transformers" --pretrained_model=SAVSNET/PetBERT --do_eval=True --dataset=PetBERT_annonymisation/datasets --text_column=sentence --label_column=entities --anonymisation=False --max_epochs=500 --batch_size=64 --learning_rate=5e-5 --hidden_size=256 --dropout=0.1 --output_dir=PetBERT_Disease/dataset/v2/out

-- Evaluating Transformers Model:
python run_model_ner.py --backend "transformers" --pretrained_model="projects/PetBERT_annonymisation/models/transformer/petbert_base_uncased" --do_eval=True --dataset="datasets/samples/n_25.csv" --text_column="narrative" --label_column="entities" --anonymisation=False --output_dir="out"

-- Annonymising a dataset:
python run_model_ner.py --backend "transformers" --pretrained_model "projects/PetBERT_annonymisation/models/transformer/petbert_base_uncased" --dataset "datasets/samples/n_25.csv" --text_column="narrative" --anonymisation True --output_dir "out"
"""

if __name__ == "__main__":
    transformer_main(args)