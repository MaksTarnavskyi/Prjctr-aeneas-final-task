#!/usr/bin/env python
"""Prepare data for transformers data loader. Save each row in the file as a dict

Usage:

    $ python utils/preprocess_data.py --source_file path/to/file.src
                           --labels_file path/to/file.lbl
                           --output_file path/to/output

"""
from helpers import remove_if_exist, split_text_into_tokens, init_logger
from tqdm.auto import tqdm
import argparse
import json

logger = init_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file',
                        help='Path to the file.src',
                        required=True)
    parser.add_argument("--labels_file",
                        help="Path to the file.lbl",
                        required=True)
    parser.add_argument("--output_file",
                        help="Path to the output file",
                        required=True)
    args = parser.parse_args()

    logger.info(f"Start data preprocessing with params: {json.dumps(args.__dict__)}")
    source_file = open(args.source_file, "r")
    labels_file = open(args.labels_file, "r")

    remove_if_exist(args.output_file)
    output_file = open(args.output_file, "a")

    for sentence, sentence_labels in tqdm(zip(source_file, labels_file)):
        tokens = split_text_into_tokens(sentence)

        labels = [int(label) for label in split_text_into_tokens(sentence_labels)]
        json_dict = json.dumps({"tokens": tokens, "labels": labels})+"\n"
        output_file.write(json_dict)

    source_file.close()
    labels_file.close()
    output_file.close()
    logger.info(f"Data preprocessed and saved in {args.output_file}")


if __name__ == '__main__':
    main()
