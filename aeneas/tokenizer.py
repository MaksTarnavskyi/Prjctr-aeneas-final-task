"""
AeneasTokenizer module for performing tokenization with transformer model's tokenizer
    and do alignment of word-level labels
"""
from transformers import AutoTokenizer
from datasets import Dataset
from utils.helpers import init_logger

logger = init_logger(__name__)


class AeneasTokenizer:
    def __init__(self, transformer_name: str, cache_dir: str = "cache/"):
        """
        Args:
            transformer_name: the name of the transformer model (in order to use the corresponding tokenizer)
            cache_dir: folder where to save loaded tokenizer
        """
        self.transformer_name = transformer_name
        self.cache_dir = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.transformer_name,
                                                       add_prefix_space=True,
                                                       cache_dir=self.cache_dir)
        self.label_all_tokens = False  # label only the first token of the word; all other word's tokens ignore
        logger.info("AeneasTokenizer initialized")

    def tokenize_and_align_labels(self, data: Dataset) -> Dataset:
        """
        Transform word-level tokens into transformer model's tokens
        As each word can consist of several tokens, based on 'label_all_tokens' flag
            we label only the first token of the word
            or we can label all word tokens (this is an area for further work)
                (in such case we should use mean or majority votes for assigning predicted label to word)
        Args:
            data: Dataset with required columns ["tokens", "labels"]

        Returns:
            tokenized_data: Dataset
        """
        tokenized_inputs = self.tokenizer(data["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(data["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        logger.info("AeneasTokenizer tokenized and aligned labels")
        return tokenized_inputs
