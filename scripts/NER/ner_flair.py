import flair
import torch
from utils.NER.flair_tools import (
    prepare_flair_dataset,
    train_flair_ner_model,
    test_set_eval,
    get_predictions,
    anonymisation,
    save_predictions,
)
from utils.dataset_tools.load import load_dataset_file


import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def train(args, corpus):
    model = train_flair_ner_model(
        corpus,
        pretrained_model=args.pretrained_model,
        model_output_path=args.output_dir,
        learning_rate=args.learning_rate,
        mini_batch_size=args.batch_size,
        max_epochs=args.epochs,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )
    test_set_eval(corpus=corpus, model=model)


def main(args):
    flair.device = torch.device(args.device)
    loaded_dataset = load_dataset_file(args.dataset)
    corpus = prepare_flair_dataset(
        loaded_dataset, text_col=args.text_column, label_col=args.entity_column
    )
    if args.mode == "train":
        logging.info("Training model...")
        train(args, corpus)
    else:
        if not args.dataset.endswith(".csv"):
            logging.info(f'{args.mode} onto the inference split')
        torch_model = torch.load(
            args.pretrained_flair_model, map_location=args.device, weights_only=False
        )
        model = flair.models.SequenceTagger.load(torch_model)
        if args.mode == "eval" and args.pretrained_flair_model:
            logging.info(f"Evaluating model from {args.pretrained_flair_model}")
            eval_report = test_set_eval(corpus=corpus, model=model)
            if args.output_dir.endswith(".csv"):
                eval_report.to_csv(f"{args.output_dir}, index=False")
            else:    
                eval_report.to_csv(f"{args.output_dir}/evaluation_report.csv, index=False")
        elif args.mode == "infer" and args.pretrained_flair_model:
            logging.info(f"Inference with model from {args.pretrained_flair_model}")
            preds = get_predictions(corpus=corpus, model=model)
            if args.anonymisation:
                preds = anonymisation(loaded_dataset["test"], preds, args.text_column)
            save_predictions(preds, args.output_dir)

        else:
            raise ValueError("Invalid mode or no args.pretrained_flair_model provided")
