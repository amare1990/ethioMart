import shap
import lime
import lime.lime_text
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from sklearn.metrics import classification_report
from datasets import load_dataset
import matplotlib.pyplot as plt


class NERModelInterpretability:
    def __init__(self, model_name='xlm-roberta-base', dataset_path='labeled_data.conll'):
        """
        Initialize the interpretability class for the NER model.
        :param model_name: Pre-trained model name for token classification (default: 'xlm-roberta-base').
        :param dataset_path: Path to the labeled dataset in CoNLL format (default: 'labeled_data.conll').
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.dataset = self.load_and_prepare_data()


    def load_and_prepare_data(self):
        """
        Load the dataset and prepare for model evaluation.
        :return: Loaded dataset for analysis.
        """
        # Load the dataset from CoNLL format
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Process the CoNLL formatted data into sentences and labels
        sentences = []
        labels = []
        sentence = []
        label = []

        for line in lines:
            if line.strip():
                token, label_token = line.strip().split()
                sentence.append(token)
                label.append(label_token)
            else:
                sentences.append(sentence)
                labels.append(label)
                sentence, label = [], []

        return {'sentences': sentences, 'labels': labels}
