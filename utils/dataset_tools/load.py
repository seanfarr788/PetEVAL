from datasets import load_from_disk, load_dataset
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] | %(message)s")


from datasets import load_from_disk, load_dataset

import logging
from pathlib import Path
from datasets import load_dataset, load_from_disk, DatasetDict
from typing import Union


def load_dataset_file(file_path: str, testing: Union[bool, int] = False) -> DatasetDict:
    """
    Load a dataset from disk or other formats using the `datasets` library.

    Parameters:
        file_path (str): Path to the dataset on disk (e.g., Arrow, CSV, JSON, Parquet) or a Hugging Face repository.
        testing (bool or int): If True, load only 50 samples per split. If an integer, load that number of samples.

    Returns:
        DatasetDict: A dictionary containing dataset splits.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"File not found: {file_path}")

    dataset = (
        _try_load_from_disk(file_path)
        or _try_load_from_file(file_path)
        or _try_load_from_repo(file_path)
    )

    if dataset is None:
        raise ValueError(f"Could not load dataset from {file_path}.")

    if testing:
        dataset = _apply_testing_environment(dataset, testing)

    return dataset


def _try_load_from_disk(file_path: Path) -> Union[DatasetDict, None]:
    try:
        dataset = load_from_disk(str(file_path))
        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({"test": dataset})
        logging.info(f"Loaded Arrow dataset with splits: {list(dataset.keys())}")
        return dataset
    except Exception as e:
        logging.warning(f"Failed to load as Arrow dataset: {e}")
        return None


def _try_load_from_file(file_path: Path) -> Union[DatasetDict, None]:
    try:
        if file_path.suffix in {".csv", ".json", ".parquet"}:
            dataset = load_dataset(
                file_path.suffix.lstrip("."), data_files=str(file_path)
            )
            if len(dataset.keys()) == 1:
                dataset = DatasetDict({"test": dataset[next(iter(dataset.keys()))]})
            logging.info(
                f"Loaded dataset ({file_path.suffix}) with splits: {list(dataset.keys())}"
            )
            return dataset
    except Exception as e:
        logging.warning(f"Failed to load as {file_path.suffix} dataset: {e}")
        return None


def _try_load_from_repo(file_path: Path) -> Union[DatasetDict, None]:
    try:
        dataset = load_dataset(str(file_path))
        logging.info(
            f"Loaded dataset from repository with splits: {list(dataset.keys())}"
        )
        return dataset
    except Exception as e:
        logging.warning(f"Failed to load from dataset repository: {e}")
        return None


def _apply_testing_environment(
    dataset: DatasetDict, sample: Union[bool, int] = 50
) -> DatasetDict:
    """Select a limited number of samples per dataset split for testing purposes."""
    if isinstance(sample, bool):
        sample = 50  # Default to 50 if True
    try:
        for split in dataset.keys():
            dataset[split] = (
                dataset[split].shuffle().select(range(sample))
                if sample
                else dataset[split]
            )
        logging.warning(
            "#### TEST MODE ENABLED: Only loading a subset of examples from each split. ####"
        )
    except Exception as e:
        logging.warning(f"Failed to apply testing environment to dataset: {e}")
    return dataset


if __name__ == "__main__":
    # Test loading a dataset from disk
    dataset = load_dataset_file(
        "projects/PetBERT_prescription/AB_labels_with_text_sample.csv"
    )
    assert isinstance(dataset, dict), "Dataset must be a dictionary"
    logging.info(f"Dataset loaded successfully: {dataset.keys()}")
