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

    def interpret_with_lime(self, sentence_idx=0):
        """
        Use LIME to explain the model's predictions for a given sentence.
        :param sentence_idx: Index of the sentence to interpret (default: 0).
        """
        # Choose a sentence from the dataset
        sentence = self.dataset['sentences'][sentence_idx]
        true_labels = self.dataset['labels'][sentence_idx]

        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        # Define a LIME text explainer
        lime_explainer = lime.lime_text.LimeTextExplainer(class_names=list(set([label for sublist in self.dataset['labels'] for label in sublist])))

        def predict_fn(texts):
            """
            Predict function for LIME: convert text to input IDs and get predictions.
            :param texts: List of text sentences.
            :return: List of model predictions (probabilities).
            """
            tokenized_inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
            logits = self.model(**tokenized_inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            return probabilities.detach().numpy()

        # Generate explanation using LIME
        explanation = lime_explainer.explain_instance(' '.join(sentence), predict_fn, num_features=10)

        # Display explanation
        explanation.show_in_notebook(text=True)

        # Print the true labels for analysis
        print(f"True labels: {true_labels}")

    def analyze_difficult_cases(self):
        """
        Analyze difficult cases where the model struggles to identify entities correctly.
        :return: List of difficult cases with predictions and true labels.
        """
        difficult_cases = []

        for idx, sentence in enumerate(self.dataset['sentences']):
            # Tokenize the sentence
            inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
            logits = self.model(**inputs).logits
            predictions = torch.argmax(logits, dim=-1)

            # Get predicted labels
            predicted_labels = [self.model.config.id2label[p.item()] for p in predictions[0]]

            # Compare with true labels
            true_labels = self.dataset['labels'][idx]

            if predicted_labels != true_labels:
                difficult_cases.append({
                    'sentence': sentence,
                    'predicted_labels': predicted_labels,
                    'true_labels': true_labels
                })

        return difficult_cases
