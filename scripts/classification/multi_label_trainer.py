import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)

from torch import nn
import numpy as np

# from sklearn.metrics import f1_score, precision_score, recall_score
# from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from utils.classification.datasets import multi_label_dataset
from utils.classification.metrics import multi_label_metrics, multi_label_evaluation
from utils.classification.training import multi_label_class_weights
from utils.training_tools import epoch_strategy, optimal_num_proc
import pandas as pd
from setfit import (
    Trainer as SetFitTrainer,
    SetFitModel,
    TrainingArguments as SetFitTrainingArguments,
)
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")


def train_transformer_model(args, dataset, id2label, label2id):
    """
    Trains a multi-label classification model using the provided dataset and arguments.

    Args:
        args (Namespace): A namespace object containing various training arguments and configurations.
        dataset (Dict[str, Dataset]): A dictionary containing the training, evaluation, and test datasets.
        id2label (Dict[int, str]): A dictionary mapping label IDs to label names.
        label2id (Dict[str, int]): A dictionary mapping label names to label IDs.

    Returns:
        None
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model,
        problem_type="multi_label_classification",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    args, callbacks = epoch_strategy(args)

    training_args = TrainingArguments(
        output_dir=args.output_dir,  # output directory
        num_train_epochs=args.epochs,  # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=args.output_dir,  # directory for storing logs
        logging_steps=1000,
        save_strategy="epoch",
        eval_strategy="epoch",
        dataloader_num_workers=optimal_num_proc(),
        dataloader_pin_memory=True,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    class CustomTrainer(Trainer):
        def __init__(self, *args, train_dataset=None, use_pos_weight=True, **kwargs):
            # Call Hugging Face Trainer with the dataset so training works
            super().__init__(*args, train_dataset=train_dataset, **kwargs)
            
            # Compute weights from the training dataset
            if train_dataset is None:
                raise ValueError("CustomTrainer requires a train_dataset for weight calculation.")

            class_weights = multi_label_class_weights(train_dataset)
            
            if use_pos_weight:
                self.loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights)
            else:
                self.loss_fct = torch.nn.BCEWithLogitsLoss(weight=class_weights, reduction="mean")
            
            self.loss_fct.to(self.model.device)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.get("labels")
            if labels is not None:
                labels = labels.float()

            outputs = model(**inputs)
            logits = outputs.logits
            loss = self.loss_fct(logits, labels)

            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        compute_metrics=multi_label_metrics,
        callbacks=callbacks,
        use_pos_weight=False  # Set to True to use positive weights for loss calculation
    )

    logging.info(f"Training model on {len(dataset["train"])} samples")
    trainer.train(args.checkpoint)
    logging.info(f"Evaluating model on {len(dataset["test"])} samples")
    predictions = trainer.predict(dataset["test"])
    if args.threshold is None:
        logging.info("No threshold provided. Defaulting to 0.5")
        args.threshold = 0.5
    report = multi_label_evaluation(
        predictions.predictions,
        predictions.label_ids,
        threshold=args.threshold,
        report=True,
    )
    logging.info(report)
    report.to_csv(args.output_dir + "classification_report.csv", index=False)


def inference_model(args, dataset):
    """
    Perform inference using a pre-trained multi-label classification model on a given dataset.

    Args:
        args (Namespace): A namespace object containing the following attributes:
            - pretrained_model (str): Path or identifier of the pre-trained model.
            - output_dir (str): Directory where the output predictions will be saved.
            - batch_size (int): Batch size for evaluation.
            - threshold (float): Threshold for multi-label classification.
        dataset (DatasetDict): A dataset dictionary containing the test dataset.

    Returns:
        None: The function saves the predictions to a CSV file in the specified output directory.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained_model,
        problem_type="multi_label_classification",
    )

    logging.info(f"Inferencing model on {dataset['test']} samples")
    config = AutoConfig.from_pretrained(args.pretrained_model)
    model.id2label = config.id2label
    model.num_labels = len(config.id2label)
    eval_args = TrainingArguments(
        output_dir="/",
        per_device_eval_batch_size=args.batch_size,
        dataloader_num_workers=optimal_num_proc(),
        dataloader_pin_memory=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
    )
    if args.threshold is None:
        logging.info("No threshold provided. Defaulting to 0.5")
        args.threshold = 0.5
    predictions = trainer.predict(dataset["test"])

    if args.mode == "infer":
        y_pred = multi_label_evaluation(
            y_pred=predictions.predictions,
            y_true=None,
            labels=config.id2label.values(),
            threshold=args.threshold,
            report=False,
        )

        original_dataset = dataset["test"].to_pandas()
        try:
            original_dataset.drop(
                columns=[
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                    "__index_level_0__",
                ],
                inplace=True,
            )
        except KeyError:
            original_dataset.drop(
                columns=["input_ids", "attention_mask", "token_type_ids"],
                inplace=True,
            )
        for idx, label in enumerate(config.id2label.values()):
            original_dataset[label] = y_pred[:, idx]
        if args.output_dir.endswith(".csv"):
            original_dataset.to_csv(args.output_dir, index=False)
        else:
            original_dataset.to_csv(args.output_dir + "predictions.csv", index=False)
    else:

        y_pred = predictions.predictions
        # apply threshold
        y_pred = (y_pred > args.threshold).astype(int)
        # convert to DataFrame
        report = multi_label_evaluation(
            y_pred=predictions.predictions,
            y_true=dataset["test"]["labels"],
            labels=config.id2label.values(),
            threshold=args.threshold,
            report=True,
        )
        report = pd.DataFrame(report).transpose()
        logging.info(report)
        if args.output_dir.endswith(".csv"):
            report.to_csv(args.output_dir, index=False)
        else:
            report.to_csv(args.output_dir + "classification_report.csv", index=False)


def train_setfit_model(args, pretrained_model, dataset, labels, output_dir="out/"):
    args, callbacks = epoch_stratergy(args)
    logging.info("Training using sentence embeddings...")
    model = SetFitModel.from_pretrained(
        pretrained_model, multi_target_strategy="one-vs-rest", labels=labels
    ).to("cuda")

    training_args = SetFitTrainingArguments(
        output_dir=output_dir,
        num_epochs=(1, args.epochs),
        # loss=CosineSimilarityLoss(model),
        batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        sampling_strategy="undersampling",  # "undersampling" or "oversampling"
        save_total_limit=1,
        # metric_for_best_model="accuracy",
        # use_amp=True,
    )
    for split in dataset.keys():
        try:
            dataset[split] = dataset[split].rename_column("labels", "label")
        except:
            pass

    trainer = SetFitTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        column_mapping={"text": "text", "label": "label"},
        # callbacks=callbacks,
    )
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)

    # def train_sentence_model(args, dataset, id2label):
    #     class MultiLabelClassifier(nn.Module):
    #         def __init__(self, base_model_name, num_labels):
    #             super(MultiLabelClassifier, self).__init__()
    #             self.sentence_transformer = SentenceTransformer(base_model_name)
    #             self.classifier = nn.Sequential(
    #                 nn.Linear(
    #                     self.sentence_transformer.get_sentence_embedding_dimension(),
    #                     num_labels,
    #                 ),
    #                 nn.Sigmoid(),
    #             )

    #         def forward(self, input_texts):
    #             embeddings = self.sentence_transformer.encode(
    #                 input_texts, convert_to_tensor=True
    #             )
    #             return self.classifier(embeddings)

    #     # Initialize model
    #     base_model_name = "projects/PetBERT_pretraining/models/sentence_transformer"
    #     model = MultiLabelClassifier(base_model_name, num_labels=len(id2label))
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model.to(device)

    #     def model_train(train_dataloader, model, optimizer, loss_fn, epoch):
    #         model.train()
    #         total_loss = 0
    #         for batch in train_dataloader:
    #             sentences = batch["text"]
    #             labels = batch["labels"].to(device)

    #             # Forward pass
    #             optimizer.zero_grad()
    #             predictions = model(sentences)
    #             loss = loss_fn(predictions, labels)

    #             # Backward pass
    #             loss.backward()
    #             optimizer.step()

    #             total_loss += loss.item()
    #             tqdm.write(
    #                 f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader):.4f}"
    #             )

    #     def model_eval(val_dataloader, model, threshold=0.5):
    #         model.eval()
    #         all_predictions = []
    #         all_labels = []
    #         with torch.no_grad():
    #             for batch in val_dataloader:
    #                 sentences = batch["sentence"]
    #                 labels = batch["labels"].to(device)
    #                 predictions = model(sentences)

    #                 all_predictions.append(predictions.cpu().numpy())
    #                 all_labels.append(labels.cpu().numpy())

    #         all_predictions = np.vstack(all_predictions)
    #         all_labels = np.vstack(all_labels)

    #         # Calculate metrics
    #         val_f1 = f1_score(
    #             all_labels, (all_predictions > threshold).astype(int), average="micro"
    #         )
    #         val_precision = precision_score(
    #             all_labels, (all_predictions > threshold).astype(int), average="micro"
    #         )
    #         val_recall = recall_score(
    #             all_labels, (all_predictions > threshold).astype(int), average="micro"
    #         )

    #         tqdm.write(
    #             f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
    #         )

    def trainer(
        model,
        train_set,
        eval_set,
        num_epochs,
        batch_size=8,
        threshold=0.5,
        output_dir="model_outputs/ICD/",
        loss_fn=nn.BCELoss(),
        optimizer=AdamW(model.parameters(), lr=5e-5),
    ):
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
        for epoch in tqdm(range(1, num_epochs), desc="Epochs"):
            model_train(train_dataloader, model, optimizer, loss_fn, epoch)
            model_eval(val_dataloader, model, threshold)
        model.sentence_transformer.save(f"{output_dir}/sentence_transformer_petbert")
        torch.save(
            model.classifier.state_dict(), f"{output_dir}/classifier_petbert.pth"
        )
        logging.info("Model saved successfully!")

    trainer(
        model,
        dataset["train"],
        dataset["eval"],
        num_epochs=5,
        batch_size=args.batch_size,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )


def main(args):
    dataset, id2label, label2id = multi_label_dataset(args)
    if args.mode == "train":
        if args.embedding_level == "word":
            logging.info("Training using word embeddings...")
            train_transformer_model(args, dataset, id2label, label2id)
            # inference_model(args, dataset)

        if args.embedding_level == "setfit":
            logging.info("Training using setfit...")
            train_setfit_model(
                args,
                pretrained_model=args.pretrained_model,
                dataset=dataset,
                labels=id2label.values(),
                output_dir=args.output_dir,
            )
        if args.embedding_level == "sentence":
            logging.info("Training using sentence embeddings...")
            train_sentence_model(
                args,
                pretrained_model=args.pretrained_model,
                dataset=dataset,
                labels=id2label.values(),
                output_dir=args.output_dir,
            )
        else:
            raise ValueError(
                f"embedding_level must be either {args.embedding_level.choices}"
            )
    else:
        inference_model(args, dataset)
