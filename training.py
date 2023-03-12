import re
import string

import tensorflow as tf


def download_file(fname: str, origin: str, untar: bool) -> str:
    """
    Downloads a file from a URL using TensorFlow's `get_file` utility function.

    Args:
        fname (str): The name of the file to download.
        origin (str): The URL to download the file from.
        untar (bool): Whether to extract the contents of a tar archive after downloading.

    Returns:
        str: The path to the downloaded file.
    """
    return tf.keras.utils.get_file(fname=fname, origin=origin, untar=untar, cache_dir=".", cache_subdir=".")


def validation_dataset_from_directory(directory: str, batch_size: int, validation_split: float, seed: int) \
        -> tf.data.Dataset:
    """
    Creates a validation dataset from the images in a directory using TensorFlow's `text_dataset_from_directory` utility function.

    Args:
        directory (str): The path to the directory containing the data.
        batch_size (int): The batch size for the dataset.
        validation_split (float): The fraction of data to use for validation.
        seed (int): The random seed for shuffling the images.

    Returns:
        any: A `tf.data.Dataset` object representing the validation dataset.
    """
    return tf.keras.utils.text_dataset_from_directory(
        directory=directory,
        batch_size=batch_size,
        validation_split=validation_split,
        subset='validation',
        seed=seed
    )


def training_dataset_from_directory(directory: str, batch_size: int, validation_split: float, seed: int) \
        -> tf.data.Dataset:
    """
    Creates a training dataset from the images in a directory using TensorFlow's `text_dataset_from_directory` utility function.

    Args:
        directory (str): The path to the directory containing the data.
        batch_size (int): The batch size for the dataset.
        validation_split (float): The fraction of data to use for validation.
        seed (int): The random seed for shuffling the images.

    Returns:
        any: A `tf.data.Dataset` object representing the training dataset.
    """
    return tf.keras.utils.text_dataset_from_directory(
        directory=directory,
        batch_size=batch_size,
        validation_split=validation_split,
        subset='training',
        seed=seed
    )


def custom_standardization(input_data: str) -> str:
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')
