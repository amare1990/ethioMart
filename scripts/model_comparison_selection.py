"""Model compariosn and selection. """

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score


class NERModelComparison:
    def __init__(self, dataset_path='../data/labeled_dataset.conll', models=None):
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


    def map_labels(self, label):
        """
        Map CoNLL labels to numerical values for token classification.
        """
        label_map = {
            'B-Product': 0, 'I-Product': 1,
            'B-LOC': 2, 'I-LOC': 3,
            'B-PRICE': 4, 'I-PRICE': 5,
            'O': 6
        }
        return label_map.get(label, 6)

    def load_data(self):
        """
        Load and process the CoNLL-formatted dataset into train/validation splits.
        """
        # Load and process the CoNLL dataset
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        data = []
        sentence, labels = [], []

        for line in lines:
            if line.strip():
                token, label = line.strip().split()
                sentence.append(token)
                labels.append(label)
            else:
                if sentence:  # End of a sentence
                    data.append({'tokens': sentence, 'labels': labels})
                    sentence, labels = [], []

        df = pd.DataFrame(data)
        df['labels'] = df['labels'].apply(lambda x: [self.map_labels(label) for label in x])

        # Split into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2)
        self.train_dataset = Dataset.from_pandas(train_df)
        self.val_dataset = Dataset.from_pandas(val_df)

    def tokenize_data(self, model_name):
        """
        Tokenize and align labels for a specific model.
        """
        def tokenize_and_align_labels(example):
            tokenized_inputs = self.tokenizer(example['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
            word_ids = tokenized_inputs.word_ids()
            aligned_labels = [-100 if word_id is None else example['labels'][word_id] for word_id in word_ids]
            tokenized_inputs['labels'] = aligned_labels
            return tokenized_inputs

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.train_dataset = self.train_dataset.map(tokenize_and_align_labels, batched=True)
        self.val_dataset = self.val_dataset.map(tokenize_and_align_labels, batched=True)