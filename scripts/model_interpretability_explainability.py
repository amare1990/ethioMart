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
