"""
AeneasPredictor module for predicting labels
"""
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer
from aeneas.tokenizer import AeneasTokenizer
from aeneas.data_loader import AeneasDataLoader
import numpy as np
from typing import List
from utils.helpers import init_logger

logger = init_logger(__name__)


class AeneasPredictor:
    def __init__(self, transformer_name: str, model_dir: str):
        """
        Args:
            transformer_name: name of transformer model
            model_dir: path to directory with saved model checkpoint
        """
        self.transformer_name = transformer_name
        self.model_dir = model_dir

        self.aeneas_tokenizer = AeneasTokenizer(transformer_name=self.transformer_name)

        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.data_collator = DataCollatorForTokenClassification(self.aeneas_tokenizer.tokenizer)
        self.aeneas_data_loader = AeneasDataLoader(aeneas_tokenizer=self.aeneas_tokenizer)
        self.aeneas_model = Trainer(
            self.model,
            data_collator=self.data_collator,
            tokenizer=self.aeneas_tokenizer.tokenizer
        )
        logger.info("AeneasPredictor initialized")

    def predict_for_file(self, source_file: str, output_file: str):
        """
        Function to load source file, predicting labels and saving output file
        Args:
            source_file: path to input source file
            output_file: path where to save output file
        """
        tokenized_dataset = self.aeneas_data_loader.load_test_file(source_file)
        logger.info("Tokenized dataset loaded")
        token_predictions, token_labels_to_ignore, _ = self.aeneas_model.predict(tokenized_dataset)
        token_predicted_labels = np.argmax(token_predictions, axis=2)  # predictions[:, :, 1]

        logger.info("Labels predicted")
        # Remove ignored index (special tokens)
        word_predicted_labels = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(token_predicted_labels, token_labels_to_ignore)
        ]

        extended_predicted_word_labels = self.aeneas_data_loader.add_missing_labels_for_long_sentences(
                                                                word_predicted_labels, tokenized_dataset)
        logger.info("Missing labels added")
        self.aeneas_data_loader.save_predicted_labels_to_file(extended_predicted_word_labels, output_file)
        logger.info(f"Labels saved in {output_file}")

    def predict_for_online(self, input_texts: List[str]):

        tokenized_dataset = self.aeneas_data_loader.load_test_sentences(input_texts)
        logger.info("Tokenized dataset loaded")
        token_predictions, token_labels_to_ignore, _ = self.aeneas_model.predict(tokenized_dataset)
        token_predicted_labels = np.argmax(token_predictions, axis=2)  # predictions[:, :, 1]

        logger.info("Labels predicted")
        # Remove ignored index (special tokens)
        word_predicted_labels = [
            [p for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(token_predicted_labels, token_labels_to_ignore)
        ]

        extended_predicted_word_labels = self.aeneas_data_loader.add_missing_labels_for_long_sentences(
                                                                word_predicted_labels, tokenized_dataset)
        logger.info("Missing labels added")
        output = self.aeneas_data_loader.return_predicted_labels(extended_predicted_word_labels)
        return output
        
