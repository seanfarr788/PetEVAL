from transformers import (
    AutoConfig,
    TrainingArguments,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    AutoTokenizer,
)
import pandas as pd

from utils.classification.metrics import multi_class_metrics, multi_class_evaluation
from utils.classification.datasets import multi_class_dataset
#from utils.classification.training import MultiClassTrainer
from transformers import Trainer
from utils.training_tools import epoch_strategy, optimal_num_proc
import logging

logging.basicConfig(level=logging.INFO)


def train(tokenized_dataset, model, tokenizer, config, args):
    args, callbacks = epoch_strategy(args)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        save_strategy="epoch",
        eval_strategy="epoch",
        dataloader_num_workers=optimal_num_proc(),
        dataloader_pin_memory=True,
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=multi_class_metrics,
        callbacks=callbacks,
    )
    logging.info("Training the model...")
    try:
        config.id2label = (
            tokenized_dataset["train"].features[args.label_column].feature.names
        )
    except:
        pass
        # config.id2label
    config.num_labels = len(set(tokenized_dataset["train"]["label"]))

    if args.checkpoint:
        logging.info(f"Resuming training from {args.checkpoint}")
        trainer.train(args.checkpoint)
    else:
        trainer.train()
    logging.info(f"Evaluating model on the validation set...")
    predictions = trainer.predict(tokenized_dataset["eval"])
    report = multi_class_evaluation(
        y_pred=predictions,
        y_true=tokenized_dataset["eval"][args.label_column],
        labels=config.id2label,
        threshold=args.threshold,
        report=True,
    )
    report = pd.DataFrame(report).transpose()
    report.to_csv(args.output_dir + "/classification_report.csv")


def predict(tokenized_dataset, model, tokenizer, args):
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            per_device_eval_batch_size=args.batch_size, output_dir="/tmp"
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    predictions = trainer.predict(tokenized_dataset["test"])
    return predictions


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.pretrained_model)
    tokenized_dataset = multi_class_dataset(
        dataset_path=args.dataset,
        text_col=args.text_column,
        tokenizer=tokenizer,
        testing=False,
    )
    config = AutoConfig.from_pretrained(args.pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model, config=config
    )

    if args.mode == "train":
        train(
            tokenized_dataset=tokenized_dataset,
            model=model,
            tokenizer=tokenizer,
            config=config,
            args=args,
        )
    else:
        predictions = predict(tokenized_dataset, model, tokenizer, args)
        if args.mode == "eval":
            logging.info("Evaluating the model on the test set...")
            report = multi_class_evaluation(
                y_pred=predictions,
                y_true=tokenized_dataset["test"][args.label_column],
                labels=config.id2label,
                threshold=args.threshold,
                report=True,
            )
            if args.output_dir.endswith(".csv"):
                report.to_csv(args.output_dir)
            else:
                report.to_csv(args.output_dir + "/classification_report.csv")
        elif args.mode == "infer":
            logging.info("Infering the model on the test set...")
            predictions = multi_class_evaluation(
                y_pred=predictions,
                y_true=None,
                labels=config.id2label,
                threshold=args.threshold,
            )
            if ".csv" in args.dataset:
                logging.info(
                    f"As the dataset is a CSV file, saving the predictions to {args.output_dir}/predictions.csv"
                )
                dataset = pd.read_csv(args.dataset)
                dataset[args.label_column] = predictions
                if ".csv" in args.output_dir:
                    dataset.to_csv(args.output_dir)
                else:
                    dataset.to_csv(args.output_dir + "/predictions.csv")
            else:
                dataset = tokenized_dataset["test"].add_column(
                    args.label_column, predictions
                )
                dataset.save_to_disk(args.output_dir)
        else:
            raise ValueError("Mode should be either train, eval or infer")
