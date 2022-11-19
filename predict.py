#!/usr/bin/env python
"""Script for generating prediction using AeneasPredictor

Usage:

    $ ./predict.py --transformer_name TRANSFORMER_NAME
                   --model_dir MODEL_DIR
                   --source_file SOURCE_FILE
                   --output_file OUTPUT_FILE

"""
import argparse
from aeneas.predictor import AeneasPredictor
import json
from utils.helpers import init_logger

logger = init_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_name',
                        help='The name of transformer model to use',
                        choices=['distilroberta-base', 'albert-base-v2', 'distilbert-base-cased',
                                 'roberta-base', 'roberta-large'], required=True)
    parser.add_argument('--model_dir',
                        help='Path where saved final model trained checkpoint', required=True)
    parser.add_argument('--source_file',
                        help='Path for source file with tokenized sentences', required=True)
    parser.add_argument('--output_file',
                        help='Path for output file where to write generated labels', required=True)

    args = parser.parse_args()
    logger.info(f"Start AeneasPredictor with params: {json.dumps(args.__dict__)}")

    model = AeneasPredictor(
        transformer_name=args.transformer_name,
        model_dir=args.model_dir
    )
    logger.info("AeneasPredictor loaded successfully")

    model.predict_for_file(args.source_file, args.output_file)
    logger.info(f"AeneasPredictor ended prediction for {args.source_file} and saved into {args.output_file}")


if __name__ == '__main__':
    main()
