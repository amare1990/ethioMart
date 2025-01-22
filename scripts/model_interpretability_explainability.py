import shap
import lime
import lime.lime_text
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import matplotlib.pyplot as plt


class NERModelInterpretability:
    def __init__(self, model_paths=None):
        """
        Initialize the interpretability class for multiple NER models.
        :param model_paths: List of paths to pre-trained models (default: three models in ../data/models_trained).
        """
        self.model_paths = model_paths or [
            "../data/models_trained/model1",
            "../data/models_trained/model2",
            "../data/models_trained/model3",
        ]
        self.models = [self.load_model(path) for path in self.model_paths]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_paths[0])

    @staticmethod
    def load_model(model_path):
        """
        Load a pre-trained token classification model from the specified path.
        :param model_path: Path to the pre-trained model.
        :return: Loaded model.
        """
        return AutoModelForTokenClassification.from_pretrained(model_path)

    def interpret_with_shap(self, sentence):
        """
        Use SHAP to interpret the models' predictions for a single sentence.
        :param sentence: The sentence to interpret.
        """
        # Tokenize the sentence
        inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

        for model in self.models:
            print(f"Interpreting model at: {model.base_model_prefix}")
            # Define a SHAP explainer
            explainer = shap.Explainer(lambda x: model(**self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)).logits, self.tokenizer)

            # Generate SHAP values for the input sentence
            shap_values = explainer([sentence])

            # Visualize SHAP values
            shap.initjs()
            shap.force_plot(shap_values[0].values, feature_names=self.tokenizer.tokenize(sentence), matplotlib=True)

    def interpret_with_lime(self, sentence):
        """
        Use LIME to explain the models' predictions for a single sentence.
        :param sentence: The sentence to interpret.
        """
        # Define a LIME text explainer
        lime_explainer = lime.lime_text.LimeTextExplainer(class_names=["Entity", "Non-Entity"])

        def predict_fn(texts):
            """
            Predict function for LIME: tokenize texts and return probabilities from all models.
            :param texts: List of text inputs.
            :return: Average predictions across all models.
            """
            model_outputs = []
            for model in self.models:
                tokenized_inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
                logits = model(**tokenized_inputs).logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                model_outputs.append(probabilities.detach().numpy())
            return np.mean(model_outputs, axis=0)

        # Generate explanation using LIME
        explanation = lime_explainer.explain_instance(sentence, predict_fn, num_features=10)

        # Display explanation
        explanation.show_in_notebook(text=True)
