from transformers import EarlyStoppingCallback
import os
import logging
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")


def optimal_num_proc():
    """
    Returns the optimal number of processes to use for data preprocessing (half the number of CPU cores)
    """
    try:
        cpu_cores = os.cpu_count()
    except:
        cpu_cores = 8
    logging.info(f"Using {int(cpu_cores/2)} processes for data preprocessing")
    return int(cpu_cores)


def epoch_strategy(args):
    """
    Set up training strategy based on the number of epochs
    If epochs is not provided, early stopping is used

    Args:
    args: dict, arguments

    Returns:
    args: dict, arguments

    """
    if args.epochs is not None:
        logging.info(f"Training for {args.epochs} epochs")
        logging.info(
            f"(reccommended) if by setting args.epochs to None, early stopping is used instead"
        )
        args.epochs = int(args.epochs)
        callbacks = []
    else:
        logging.info(f"Training with early stopping [patience: {args.patience}]")
        args.epochs = 1000
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]
    return args, callbacks


def save_strategy(batch_size, train_data_length):
    """
    If steps per epoch is greater than 250,000 save every 100,000 steps else epoch
    """
    steps_per_epoch = train_data_length / batch_size
    if steps_per_epoch > 250_000:
        save_method = "steps"
        save_steps = 100_000
    else:
        save_method = "epoch"
        save_steps = 1
    logging.info(f"Saving model every {save_steps} {save_method}")
    return save_method, save_steps


def LORA_stratergy(model, r=8, lora_alpha=16, lora_dropout=0.1, checkpoint=None):
    if checkpoint:
        logging.info(f"Loading LoRA config from checkpoint")
        model = PeftModel.from_pretrained(model, checkpoint + "/adaptor_config.json")
    else:
        logging.info(
            f"Using LoRA with r={r}, alpha={lora_alpha}, dropout={lora_dropout}"
        )
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Task type for causal language modeling
            inference_mode=False,  # We're in training mode
            r=r,  # Low-rank parameter (default is 8)
            lora_alpha=lora_alpha,  # Scaling factor
            lora_dropout=lora_dropout,  # Dropout for LoRA layers
        )
        # Wrap the model with LoRA
        model = get_peft_model(model, lora_config)

    return model
