"""Model compariosn and selection. """

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score


class NERModelComparison:
    def __init__(self, dataset_path='labeled_data.conll', models=None):
        """
        Initialize the model comparison class with dataset path and model options.
        :param dataset_path: Path to the labeled dataset in CoNLL format.
        :param models: List of model names to compare (default: multilingual NER models).
        """
        self.dataset_path = dataset_path
        self.models = models or ['xlm-roberta-base', 'distilbert-base-multilingual-cased', 'bert-base-multilingual-cased']
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.model_performance = {}
