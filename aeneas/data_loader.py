"""
Module for loading, preprocessing and saving data which is needed for model training and testing
"""
from datasets import load_dataset, load_from_disk, Dataset
from utils.helpers import read_tokenized_lines, write_lines, init_logger, make_world_level_tokens
import os
from aeneas.tokenizer import AeneasTokenizer
from typing import List

logger = init_logger(__name__)


class AeneasDataLoader:
    def __init__(self,  aeneas_tokenizer: AeneasTokenizer, cache_dir: str = "cache/"):
        """
        Args:
            aeneas_tokenizer: the instance of custom tokenizer class AeneasTokenizer
                                using which we tokenize data with transformer model's tokenizer
            cache_dir: folder where to save cached processed data
        """
        self.aeneas_tokenizer = aeneas_tokenizer
        self.transformer_name = self.aeneas_tokenizer.transformer_name
        self.cache_dir = cache_dir

        logger.info("AeneasDataLoader initialized")

    def load_tokenized_train_data(self, train_set: str, dev_set: str) -> Dataset:
        """
        Function which download preprocessed data for model training
            and tokenize it with transformer model's tokenizer
            If cache is available, it load dataset from cache

        Args:
            train_set: path to train file
            dev_set: path to dev file

        train_set and dev_set should be generated using script ./utils/preprocess_data.py
        Returns:
            tokenized_datasets : Dataset
        """
        tokenized_dataset_name_hash = str(hash("_".join([self.transformer_name, train_set, dev_set])))
        tokenized_dataset_save_path = os.path.join(self.cache_dir, tokenized_dataset_name_hash)

        if os.path.isdir(tokenized_dataset_save_path):
            logger.info("Start loading tokenized dataset from cache")
            tokenized_datasets = load_from_disk(tokenized_dataset_save_path)

        else:
            logger.info("Start loading dataset")
            datasets = load_dataset('json',
                                    data_files={"train": train_set,
                                                "val": dev_set},
                                    cache_dir=self.cache_dir)

            logger.info("Start tokenizing dataset")
            tokenized_datasets = datasets.map(self.aeneas_tokenizer.tokenize_and_align_labels, batched=True)
            tokenized_datasets.save_to_disk(tokenized_dataset_save_path)

        logger.info("Tokenized dataset loaded and ready to use")
        return tokenized_datasets

    def load_test_file(self, source_file: str) -> Dataset:
        """
        Function that download source_file for testing adn tokenize it.

            As each transformer model has limit for maximal token sequence length, that model can process in one time,
            we need to remember the initial amount of word_level tokens in the sentences
            For words which will not have labels, we in future extend labels with zeros to be able evaluate our file

            Also we generate pseudo labels to understand which predicted token labels in future we need to skip,
            and which to use (skip labels for SEP token and so on)
            (In order to use function for tokenization and alignment as we used for training data)

        Args:
            source_file: str path to source file (we don't need to preprocess test file, just use raw source txt file)

        Returns:
            Dataset
        """
        word_level_tokens = read_tokenized_lines(source_file)
        logger.info("Loaded source file for prediction")
        word_level_token_counts = [len(tokens_in_line) for tokens_in_line in word_level_tokens]
        default_labels = [[0]*token_count for token_count in word_level_token_counts]

        dataset_dict = {"tokens": word_level_tokens,
                        "labels": default_labels,
                        "tokens_count": word_level_token_counts
                        }

        dataset = Dataset.from_dict(dataset_dict)
        logger.info("Dataset object created")

        tokenized_dataset = dataset.map(self.aeneas_tokenizer.tokenize_and_align_labels, batched=True)
        logger.info("Dataset tokenized")

        return tokenized_dataset

    @staticmethod
    def add_missing_labels_for_long_sentences(word_predicted_labels: List[List[int]],
                                              tokenized_dataset: Dataset) -> List[List[int]]:
        """
        Function for adding missing labels to sentences which have amount of tokens more than
            transformer model's limit for maximal tokens sequence length
        Args:
            word_predicted_labels: List[List[int]]
            tokenized_dataset: Dataset which has column "tokens_count"

        Returns:
            List[List[int]] - list of extended labels
        """
        extended_predicted_word_labels = []
        count_of_extended_lines = 0
        for i, (predicted_labels, original_word_tokens_count) in enumerate(zip(word_predicted_labels,
                                                                               tokenized_dataset['tokens_count'])):
            count_predicted_labels = len(predicted_labels)
            if count_predicted_labels != original_word_tokens_count:
                added_labels = [0] * (original_word_tokens_count - count_predicted_labels)
                predicted_labels.extend(added_labels)
                count_of_extended_lines += 1
            extended_predicted_word_labels.append(predicted_labels)
        logger.info(f"Extended {count_of_extended_lines} lines with missing labels")

        return extended_predicted_word_labels

    @staticmethod
    def save_predicted_labels_to_file(extended_predicted_word_labels: List[List[int]], output_file: str):
        """
        Save labels in output file
        Args:
            extended_predicted_word_labels: list of predicted labels
            output_file: path to output file where to save labels
        """
        output = [" ".join(list(map(str, pred_labels))) for pred_labels in extended_predicted_word_labels]
        write_lines(output, output_file)
        logger.info(f"Saved labels in {output_file}")

    @staticmethod
    def return_predicted_labels(extended_predicted_word_labels: List[List[int]]):
        output = [" ".join(list(map(str, pred_labels))) for pred_labels in extended_predicted_word_labels]
        return output

    def load_test_sentences(self, texts: List[str]):
        logger.info(f"Received {len(texts)} for prediction")

        word_level_tokens = make_world_level_tokens(texts)

        logger.info("Loaded source file for prediction")
        word_level_token_counts = [len(tokens_in_line) for tokens_in_line in word_level_tokens]
        default_labels = [[0]*token_count for token_count in word_level_token_counts]

        dataset_dict = {"tokens": word_level_tokens,
                        "labels": default_labels,
                        "tokens_count": word_level_token_counts
                        }

        dataset = Dataset.from_dict(dataset_dict)
        logger.info("Dataset object created")

        tokenized_dataset = dataset.map(self.aeneas_tokenizer.tokenize_and_align_labels, batched=True)
        logger.info("Dataset tokenized")

        return tokenized_dataset

