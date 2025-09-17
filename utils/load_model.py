import logging
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    BertTokenizerFast,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s:%(lineno)d] - %(message)s",
)
import torch


def import_model(
    task_type: str,
    model_path: str,
    tokenizer_path: str = None,
    flash_attention: bool = False,
    dtype: str = "float16",
    random_weights: bool = False,
):
    tokenizer = get_tokenizer(model_path, tokenizer_path)
    model = get_model(
        model_path=model_path,
        task_type=task_type,
        dtype=dtype,
        random_weights=random_weights,
    )
    if flash_attention:
        model = enable_flash_attention(model)
    logging.info(f"Using {dtype} for model weights")
    return model, tokenizer


def get_tokenizer(model_path: str, tokenizer_path: str = None) -> AutoTokenizer:
    """
    Loads a tokenizer from the specified path.

    Args:
        model_path (str): Path to the model directory.
        tokenizer_path (str, optional): Path to the tokenizer directory. If None, defaults to model_path.

    Returns:
        PreTrainedTokenizer: The loaded tokenizer.

    Raises:
        ValueError: If the tokenizer_path is different from model_path and may cause issues.

    Notes:
        If tokenizer_path is not provided, it defaults to model_path. A warning is logged if the tokenizer_path
        is different from the model_path, as this may cause issues.
    """
    if tokenizer_path is None:
        logging.info(f"No tokenizer path provided. Using model path {model_path}")
        tokenizer_path = model_path
    if tokenizer_path != model_path:
        logging.warning(
            "The tokenizer and the model are different. This may cause issues."
        )
    logging.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = BertTokenizerFast(vocab_file="pretraining/tokenizer/vocab.txt")
    return tokenizer


def get_model(model_path, task_type, dtype, random_weights=False):
    """
    Load or initialize a model based on the specified task type and data type.

    Args:
        model_path (str): Path to the model or model configuration.
        task_type (str): Type of task for the model. Can be "masked" for masked language modeling
                         or "causal" for causal language modeling.
        dtype (torch.dtype): Data type for the model's weights.
        load_pretrained (bool): If True, loads pretrained weights; if False, initializes randomly.

    Returns:
        PreTrainedModel: The loaded or initialized model.
    """
    action = "Loading" if random_weights == False else "Initializing"
    logging.info(
        f"{action} {'pretrained' if random_weights==False else 'random'} model using {model_path}"
    )

    config = AutoConfig.from_pretrained(model_path)
    config.torch_dtype = dtype

    if task_type == "masked":
        model = (
            AutoModelForMaskedLM.from_pretrained(
                model_path, config=config, torch_dtype=dtype
            )
            if random_weights == False
            else AutoModelForMaskedLM.from_config(config)
        )
    elif task_type == "causal":
        model = (
            AutoModelForCausalLM.from_pretrained(
                model_path, config=config, torch_dtype=dtype
            )
            if random_weights == False
            else AutoModelForCausalLM.from_config(config)
        )
    else:
        raise ValueError(
            f"Unsupported task_type '{task_type}'. Choose 'masked' or 'causal'."
        )
    if action == "Initializing":
        logging.info(
            f"Model initialized with random weights. Model config: {model.config}"
        )
    return model


def enable_flash_attention(model):
    """
    Enables Flash Attention for the given model if the flash_attn module is installed.

    This function attempts to import the flash_attn module and replace the
    nn.MultiheadAttention layers in the model with FlashAttention. If the
    flash_attn module is not installed, it logs an error message and returns
    the original model. If any other exception occurs during the process, it
    logs the error message.

    Args:
        model (torch.nn.Module): The model for which Flash Attention should be enabled.

    Returns:
        torch.nn.Module: The model with Flash Attention enabled, or the original
        model if flash_attn is not installed or an error occurs.
    """
    try:
        from flash_attn import FlashAttention
    except ImportError:
        logging.error("Flash Attention not installed. Please install flash_attn.")
        return model
    try:
        for module in model.modules():
            if isinstance(module, nn.MultiheadAttention):
                module.attention = FlashAttention()
        logging.info(f"Using Flash Attention")
    except Exception as e:
        logging.error(f"Error enabling Flash Attention: {e}")
    return model
