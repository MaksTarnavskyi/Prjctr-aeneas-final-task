#!/usr/bin/env python
"""Script for ensemble

Usage:

    $ python utils/ensemble.py --input_files  INPUT_FILES [INPUT_FILES ...]
                               --output_file OUTPUT_FILE
                               --min_count MIN_COUNT

"""
from helpers import read_tokenized_lines, write_lines, init_logger
import argparse
from typing import List

logger = init_logger(__name__)


def ensemble_labels(labels: List[List[List[str]]], min_count: int = 2) -> List[List[str]]:
    """
    For each label we count how many "1" values it have
    If count of "1" is more than min_count
        => then we assign "1" for label
        otherwise, we assign "0" for label
    Args:
        labels: List[List[List[str]]] of labels from each file and each line
        min_count: minimal amount of votes for "1" label assignment

    Returns:
        List[List[str]] - result labels based on voting
    """
    n_files = len(labels)
    n_lines = len(labels[0])
    res_labels = []
    for line_index in range(n_lines):
        line_labels = []
        n_labels = len(labels[0][line_index])
        for label_index in range(n_labels):
            count_ones = 0
            for file_index in range(n_files):
                count_ones += int(labels[file_index][line_index][label_index])
            if count_ones >= min_count:
                line_labels.append("1")
            else:
                line_labels.append("0")
        res_labels.append(line_labels)
    return res_labels


def load_files(input_filenames: List[str]) -> List[List[List[str]]]:
    loaded_files = [read_tokenized_lines(filename) for filename in input_filenames]
    return loaded_files


def ensemble(input_filenames: List[str], output_filename: str, min_count: int = 2):
    """
    Function that reads input files with labels,
        based on majority votes decided which label to token use
        and save them into output file
    Args:
        input_filenames: paths to input files
        output_filename: path where to save output
        min_count: minimal amount of votes for label assignment
    """
    input_labels = load_files(input_filenames)
    output_labels = ensemble_labels(input_labels, min_count)
    output_lines = [" ".join(labels) for labels in output_labels]
    write_lines(output_lines, output_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files",
                        help="Paths to the input files",
                        nargs='+',
                        required=True)
    parser.add_argument("--output_file",
                        help="Path where to save output file",
                        required=True)
    parser.add_argument("--min_count",
                        help="Minimal count for majority voting",
                        type=int,
                        default=2)
    args = parser.parse_args()
    ensemble(input_filenames=args.input_files,
             output_filename=args.output_file,
             min_count=args.min_count)


if __name__ == '__main__':
    main()
