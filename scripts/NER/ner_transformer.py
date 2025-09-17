import logging
import os
import glob
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification,
)

from utils.NER.preprocess import create_label_mapping, tokenize_and_align_labels
from utils.NER.evaluate import (
    compute_metrics,
    extract_entities,
    evaluate,
)
from utils.training_tools import epoch_strategy, optimal_num_proc
from utils.dataset_tools.load import load_dataset_file

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")


def train(args, tokenized_dataset, model, tokenizer, id_to_label):

    data_collator = DataCollatorForTokenClassification(tokenizer)
    args, callbacks = epoch_strategy(args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        logging_dir="./logs",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        eval_on_start=True,
        num_train_epochs=args.epochs,
        load_best_model_at_end=True,
        report_to="wandb",
        save_total_limit=3,
        seed=42,
    )

    def compute_metrics_wrapper(p):
        return compute_metrics(p, id_to_label)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"],
        compute_metrics=compute_metrics_wrapper,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=args.checkpoint if args.checkpoint else None)
    metrics = trainer.evaluate()
    print(f"Running evaluation on the test set")
    test_metrics = trainer.evaluate(tokenized_dataset["test"])
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)


def inference(
    args,
    tokenized_dataset,
    model,
    tokenizer,
    config,
    eval_mode=False,
    batch_size=100_000,
):
    """
    Perform inference on the test split of the dataset.
    If the dataset is large, the predictions are checkpointed every `batch_size` samples.

    Parameters:
        args (argparse.Namespace): Arguments passed to the script.
        tokenized_dataset (datasets.DatasetDict): Tokenized dataset containing test split.
        model (transformers.PreTrainedModel): Pretrained model for inference.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
        config (transformers.PretrainedConfig): Model configuration.
        eval_mode (bool): If True, evaluate the model predictions.
        batch_size (int): Number of samples to process in each batch.

    Returns:
        None (saves the predictions to disk).
    """
    logging.info(f"Performing {args.mode} on dataset test split")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="/", report_to="none"),
    )

    dataset = tokenized_dataset["test"]
    num_samples = len(dataset)

    if num_samples <= 100_000:
        predictions = trainer.predict(dataset)

        if eval_mode:
            evaluate(tokenized_dataset, predictions, config, args.output_dir)
        else:
            dataset = extract_entities(
                dataset=dataset,
                predictions=predictions,
                tokenizer=tokenizer,
                id2label=config.id2label,
            )
            df = dataset.to_pandas()[[args.text_column, "new_entities"]]
            output_path = (
                args.output_dir
                if args.output_dir.endswith(".csv")
                else f"{args.output_dir}/transformers_preds.csv"
            )
            df.to_csv(output_path, index=False)
        return
    logging.info(f"Large dataset detected, checkpointing every {batch_size} samples")
    if args.output_dir.endswith(".csv"):
        cache_dir = args.output_dir.replace(".csv", "")
    else:
        cache_dir = args.output_dir
    os.makedirs(cache_dir, exist_ok=True)

    num_batches = (num_samples + batch_size - 1) // batch_size
    cached_files = []

    # Resume from last checkpoint if available
    try:
        existing_checkpoints = sorted(glob.glob(f"{args.output_dir}/checkpoint_*.csv"))
        start_idx = (
            len(existing_checkpoints) * batch_size if existing_checkpoints else 0
        )
        logging.info(f"Resuming from checkpoint {start_idx}")
    except:
        start_idx = 0

    for i in tqdm(
        range(start_idx, num_samples, batch_size),
        desc="Processing batches",
        total=num_batches,
    ):
        batch = dataset.select(range(i, min(i + batch_size, num_samples)))
        predictions = trainer.predict(batch)

        if not eval_mode:
            batch_results = extract_entities(
                dataset=batch,
                predictions=predictions,
                tokenizer=tokenizer,
                id2label=config.id2label,
            )
            df = batch_results.to_pandas()[[args.text_column, "new_entities"]]
            checkpoint_path = f"{cache_dir}/checkpoint_{i}.csv"
            df.to_csv(checkpoint_path, index=False)
            cached_files.append(checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")

    if not eval_mode:
        # Combine all cached files into a final output
        final_output_path = (
            args.output_dir
            if args.output_dir.endswith(".csv")
            else f"{args.output_dir}/transformers_preds.csv"
        )
        all_dfs = [
            pd.read_csv(file)
            for file in sorted(glob.glob(f"{args.output_dir}/checkpoint_*.csv"))
        ]
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(final_output_path, index=False)

        # Clean up checkpoint files
        for file in cached_files:
            os.remove(file)
        logging.info(
            f"Final output saved: {final_output_path}, checkpoint files removed."
        )

    if eval_mode:
        evaluate(tokenized_dataset, predictions, config, args.output_dir)


def main(args):
    dataset = load_dataset_file(args.dataset, testing=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if args.mode == "train":
        label_to_id = create_label_mapping(dataset["test"])
        id_to_label = {v: k for k, v in label_to_id.items()}
        model = AutoModelForTokenClassification.from_pretrained(
            args.pretrained_model,
            num_labels=len(label_to_id),
            id2label=id_to_label,
            label2id=label_to_id,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(args.pretrained_model)
        config = AutoConfig.from_pretrained(args.pretrained_model)
        label_to_id = config.label2id

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples=examples,
            tokenizer=tokenizer,
            label_to_id=label_to_id,
            max_length=args.context_length,
            text_column=args.text_column,
            label_column=args.entity_column,
            mode=args.mode,
        ),
        batched=True,
        num_proc=optimal_num_proc(),
    )

    if args.mode == "train":
        train(args, tokenized_dataset, model, tokenizer, id_to_label)
    elif args.mode == "eval":
        inference(args, tokenized_dataset, model, tokenizer, config, eval_mode=True)
    elif args.mode == "infer":
        inference(args, tokenized_dataset, model, tokenizer, config, eval_mode=False)
    else:
        raise ValueError("Invalid mode. Choose from 'train', 'eval', or 'infer'.")
