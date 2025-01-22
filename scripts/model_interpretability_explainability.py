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

    def interpret_with_shap(self, sentence_idx=0):
        """
        Use SHAP to interpret the model's predictions for a given sentence.
        :param sentence_idx: Index of the sentence to interpret (default: 0).
        """
        # Choose a sentence from the dataset
        sentence = self.dataset['sentences'][sentence_idx]
        true_labels = self.dataset['labels'][sentence_idx]

        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        # Define a SHAP explainer for the model
        explainer = shap.Explainer(self.model, self.tokenizer)

        # Generate SHAP values for the sentence
        shap_values = explainer(inputs['input_ids'])

        # Visualize SHAP values (displaying how much each token contributed to the classification)
        shap.initjs()
        shap.force_plot(shap_values[0].values, feature_names=sentence, matplotlib=True)

        # Print the SHAP values and the true labels for analysis
        print(f"True labels: {true_labels}")
        print(f"SHAP values for the sentence: {shap_values}")
