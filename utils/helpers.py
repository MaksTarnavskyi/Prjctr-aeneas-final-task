import yaml
from typing import List, Dict
import os
import logging
import sys


def init_logger(name: str):
    """
    Setup logger that output info into stdout
    Args:
        name: name of the logger (can be name of the file from which init logger)

    Returns:
        Logger that can output into stdout
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


logger = init_logger(__name__)


def load_yaml(filename: str) -> Dict:
    """
    Args:
        filename: path to yml file
    Returns:
        Dict generated from yml file
    """
    with open(filename) as file:
        dict_from_yml = yaml.load(file, Loader=yaml.FullLoader)
    return dict_from_yml

def make_world_level_tokens(texts: List[str]):
    text_lines = [split_text_into_tokens(text) for text in texts]
    return text_lines

def read_tokenized_lines(filename: str):
    """
    Args:
        filename: path to file
    Returns:
        list of sentences
    """
    try:
        text_lines = []
        for text_line in open(filename, "r") :
            text_lines.append(split_text_into_tokens(text_line))
        return text_lines
    except Exception as e:
        logger.error(e)
        return []


def write_lines(text_lines: List[str], output_file: str):
    """
    Args:
        text_lines: text lines to save into the file
        output_file: path to file where to save lines
    """
    text_lines = [text + "\n" for text in text_lines]
    with open(output_file, "w") as f:
        f.writelines(text_lines)


def split_text_into_tokens(text: str) -> List[str]:
    """
    Split tokenized text into tokens. Split by whitespaces
    Ex. "hello world" -> ["hello", "world"]
    Args:
        text: text to split
    Returns:
        list of tokens
    """
    return text.strip().split()


def remove_if_exist(path: str):
    """
    Args:
        path: path to file which check if exist and if exist - remove
    """
    if os.path.exists(path):
        os.remove(path)
