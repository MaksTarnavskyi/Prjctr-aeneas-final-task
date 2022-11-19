#!/usr/bin/env python
"""Script for splitting data on train and val parts

The simplest way to split data is to use 'sklearn.model_selection.train_test_split' function
However, in case when we have a huge amount of data and we don't have an enough RAM memory,
we can process data line by line and with certain probability assign it into the train or validation

Usage:

    $ python utils/split_train_val.py --input_filename INPUT_FILENAME
                                      --output_train_filename OUTPUT_TRAIN_FILENAME
                                      --output_val_filename OUTPUT_VAL_FILENAME
                                      --validation_size_rate FLOAT_NUMBER

"""
from helpers import init_logger
import argparse
from tqdm.auto import tqdm
import random
import json

logger = init_logger(__name__)


def get_random_value(val_probability: float) -> int:
    """
    This function returns with certain probability 0 or 1 values
    0 - corresponds to train
    1 - corresponds to validation
    Args:
        val_probability: probability to get value 1, probability should be in [0, 1] range
    """
    train_probability = 1 - val_probability
    return random.choices(population=[0, 1], weights=[train_probability, val_probability], k=1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_filename',
                        help='Path to the input file',
                        required=True)
    parser.add_argument("--validation_size_rate",
                        help="expected rate for validation part",
                        default=0.02)
    parser.add_argument("--output_train_filename",
                        help="Path to the output train file",
                        required=True)
    parser.add_argument("--output_val_filename",
                        help="Path to the output val file",
                        required=True)
    parser.add_argument("--random_seed",
                        help="random seed",
                        default=1)

    args = parser.parse_args()
    logger.info(f"Start splitting data on train val parts with params: {json.dumps(args.__dict__)}")
    input_file = open(args.input_filename, "r")

    output_train = open(args.output_train_filename, "w")
    output_val = open(args.output_val_filename, "w")
    output_train.close()
    output_val.close()

    output_train = open(args.output_train_filename, "a")
    output_val = open(args.output_val_filename, "a")

    random.seed(args.random_seed)
    count_train_lines = 0
    count_val_lines = 0

    for line in tqdm(input_file):
        if get_random_value(args.validation_size_rate) == 1:
            output_val.write(line)
            count_val_lines += 1
        else:
            output_train.write(line)
            count_train_lines += 1

    input_file.close()
    output_train.close()
    output_val.close()
    logger.info("Finished splitting into train and val parts")
    logger.info(f"Train: {count_train_lines}")
    logger.info(f"Validation: {count_val_lines}")
    logger.info(f"Ration: {count_val_lines/(count_val_lines+count_train_lines)} ")


if __name__ == '__main__':
    main()
