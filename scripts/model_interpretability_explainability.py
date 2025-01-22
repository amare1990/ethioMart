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

    def analyze_single_sentence(self, sentence):
        """
        Analyze a single sentence with SHAP and LIME for interpretability.
        :param sentence: The input sentence to analyze.
        """
        print("=== SHAP Analysis ===")
        self.interpret_with_shap(sentence)

        print("\n=== LIME Analysis ===")
        self.interpret_with_lime(sentence)

    def analyze_difficult_cases(self, sentences, true_labels):
        """
        Analyze difficult cases where the models struggle to identify entities correctly.
        :param sentences: List of sentences to analyze.
        :param true_labels: Corresponding ground truth labels for the sentences.
        :return: A dictionary of difficult cases for each model.
        """
        difficult_cases = {model.base_model_prefix: [] for model in self.models}

        for idx, sentence in enumerate(sentences):
            for model in self.models:
                # Tokenize the sentence
                inputs = self.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
                logits = model(**inputs).logits
                predictions = torch.argmax(logits, dim=-1)

                # Map predicted indices to labels
                predicted_labels = [model.config.id2label[token.item()] for token in predictions[0]]

                # Compare predictions with true labels
                if predicted_labels != true_labels[idx]:
                    difficult_cases[model.base_model_prefix].append({
                        'sentence': sentence,
                        'predicted_labels': predicted_labels,
                        'true_labels': true_labels[idx]
                    })

        return difficult_cases

    def generate_report(self, sentences, true_labels):
        """
        Generate a detailed report summarizing model performance and decision-making.
        :param sentences: List of sentences to analyze.
        :param true_labels: Corresponding ground truth labels for the sentences.
        :return: A dictionary summarizing the interpretability report.
        """
        # Analyze difficult cases
        difficult_cases = self.analyze_difficult_cases(sentences, true_labels)

        # Summarize results
        report = {
            'model_performance': [],
            'difficult_cases': {},
        }

        for model in self.models:
            model_name = model.base_model_prefix
            num_difficult_cases = len(difficult_cases[model_name])
            total_cases = len(sentences)
            report['model_performance'].append({
                'model_name': model_name,
                'accuracy': f"{(1 - num_difficult_cases / total_cases) * 100:.2f}%",
                'total_cases': total_cases,
                'difficult_cases': num_difficult_cases,
            })
            report['difficult_cases'][model_name] = difficult_cases[model_name]

        # Example SHAP and LIME placeholders
        report['shap_analysis'] = "SHAP results for selected sentences..."
        report['lime_analysis'] = "LIME results for selected sentences..."

        # Display the report
        print("=== Interpretability Report ===")
        for model_perf in report['model_performance']:
            print(f"Model: {model_perf['model_name']}")
            print(f"Accuracy: {model_perf['accuracy']}")
            print(f"Difficult Cases: {model_perf['difficult_cases']}/{model_perf['total_cases']}\n")

        if any(report['difficult_cases'].values()):
            print("=== Difficult Cases ===")
            for model_name, cases in report['difficult_cases'].items():
                if cases:
                    print(f"Model: {model_name}")
                    for case in cases:
                        print(f"Sentence: {' '.join(case['sentence'])}")
                        print(f"Predicted Labels: {case['predicted_labels']}")
                        print(f"True Labels: {case['true_labels']}")
                        print()
        else:
            print("No difficult cases identified.")

        return report
