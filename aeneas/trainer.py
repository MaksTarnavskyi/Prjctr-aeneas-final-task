"""
AeneasTrainer module for model training
"""
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from aeneas.tokenizer import AeneasTokenizer
from aeneas.metrics import compute_metrics
from aeneas.data_loader import AeneasDataLoader
from utils.helpers import load_yaml
from utils.helpers import init_logger

logger = init_logger(__name__)


class AeneasTrainer:
    def __init__(self, transformer_name: str,
                 train_set: str,
                 dev_set: str,
                 model_dir: str,
                 train_config_path: str,
                 pretrain_checkpoint: str = "",
                 cache_dir: str = "cache/"
                 ):
        """
        Args:
            transformer_name: name of transformer model
            train_set: path to train file
            dev_set: path to dev file
            model_dir: path to folder where to save model checkpoints
            train_config_path: path to the config with training params
            pretrain_checkpoint: path to the folder with pretrained checkpoint
            cache_dir: folder where to save transformers cache
        """
        self.transformer_name = transformer_name
        self.train_set = train_set
        self.dev_set = dev_set
        self.model_dir = model_dir
        self.pretrain_checkpoint = pretrain_checkpoint
        self.cache_dir = cache_dir
        self.train_config_path = train_config_path

        self.aeneas_tokenizer = AeneasTokenizer(transformer_name=self.transformer_name)

        if self.pretrain_checkpoint != "":
            self.model = AutoModelForTokenClassification.from_pretrained(self.pretrain_checkpoint)
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(self.transformer_name, num_labels=2)

        self.data_collator = DataCollatorForTokenClassification(self.aeneas_tokenizer.tokenizer)
        self.train_config = load_yaml(self.train_config_path)
        self.train_config['output_dir'] = self.model_dir

        self.aeneas_data_loader = AeneasDataLoader(aeneas_tokenizer=self.aeneas_tokenizer,
                                                   cache_dir=self.cache_dir
                                                   )
        logger.info("AeneasTrainer initialized")

    def train(self):
        """
        First load tokenized dataset, then initialize Trainer and start or continue model training
        """
        tokenized_datasets = self.aeneas_data_loader.load_tokenized_train_data(
            train_set=self.train_set,
            dev_set=self.dev_set
        )
        args = TrainingArguments(
            **self.train_config
        )
        print(tokenized_datasets)

        trainer = Trainer(
            self.model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["val"],
            data_collator=self.data_collator,
            tokenizer=self.aeneas_tokenizer.tokenizer,
            compute_metrics=compute_metrics
        )

        if self.pretrain_checkpoint is not None:
            logger.info(f"Continue training based on {self.pretrain_checkpoint}")
            trainer.train(self.pretrain_checkpoint)
        else:
            logger.info("Start training")
            trainer.train()
        logger.info("AeneasTrainer ended training")
