#!/usr/bin/env python
"""Script for model training

Usage:

    $ python train.py --transformer_name TRANSFORMER_NAME
                 --train_set PATH_TO_TRAIN
                 --dev_set PATH_TO_DEV
                 --model_dir PATH_TO_MODEL_DIR

"""
import argparse
import json
from aeneas.trainer import AeneasTrainer
from utils.helpers import init_logger

logger = init_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_name',
                        help='The name of transformer model to use',
                        choices=['distilroberta-base', 'albert-base-v2', 'distilbert-base-cased',
                                 'roberta-base', 'roberta-large'], required=True)
    parser.add_argument('--train_set',
                        help='Path to the train data', required=True)
    parser.add_argument('--dev_set',
                        help='Path to the dev data', required=True)
    parser.add_argument('--model_dir',
                        help='Path where to save training checkpoints', required=True)
    parser.add_argument('--pretrain_checkpoint',
                        help='Path to the pretrained checkpoint',
                        default='')
    parser.add_argument('--train_config_path',
                        help='Path to the config with training params',
                        default='train_configs/default_train_args.yml')
    parser.add_argument('--cache_dir',
                        help='Folder where to save transformers cache',
                        default='cache/')

    args = parser.parse_args()

    logger.info(f"Start AeneasTrainer with params: {json.dumps(args.__dict__)}")

    trainer = AeneasTrainer(
        transformer_name=args.transformer_name,
        train_set=args.train_set,
        dev_set=args.dev_set,
        model_dir=args.model_dir,
        train_config_path=args.train_config_path,
        pretrain_checkpoint=args.pretrain_checkpoint,
        cache_dir=args.cache_dir
    )
    logger.info("AeneasTrainer loaded successfully")

    trainer.train()
    logger.info("AeneasTrainer finished training")


if __name__ == '__main__':
    main()
