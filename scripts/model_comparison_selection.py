"""Model compariosn and selection. """

import time
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
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
        self.tokenized_data = {'train': None, 'val': None}
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
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        self.train_dataset = Dataset.from_pandas(train_df)
        self.val_dataset = Dataset.from_pandas(val_df)

    def tokenize_data(self, model_name):
        """
        Tokenize and align labels for a specific model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_and_align_labels(examples):
            """
            Tokenize the input sentence and align the labels with the tokenized text.
            """
            tokenized_inputs = self.tokenizer(
                examples['tokens'], truncation=True, padding='max_length', is_split_into_words=True
            )

            all_ner_tags = []
            for i, labels in enumerate(examples['labels']):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get word ids for each token
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        # Handle cases where word_idx might be out of range
                        if word_idx < len(labels):
                            label_ids.append(labels[word_idx])
                        else:
                            label_ids.append(self.map_labels('O'))  # Assign 'O' if out of range
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                all_ner_tags.append(label_ids)

            tokenized_inputs['labels'] = all_ner_tags
            return tokenized_inputs

        # Apply tokenization and label alignment
        self.train_dataset = self.train_dataset.map(tokenize_and_align_labels, batched=True)
        self.val_dataset = self.val_dataset.map(tokenize_and_align_labels, batched=True)

        # After tokenizing
        self.tokenized_data['train'] = self.train_dataset
        self.tokenized_data['val'] = self.val_dataset


    def get_tokenized_data(self):
        """
        Return the tokenized data for interpretability.
        """
        return self.tokenized_data


    def load_model(self, model_name):
        """
        Load the pre-trained model for token classification.
        """
        return AutoModelForTokenClassification.from_pretrained(model_name, num_labels=7)

    def train_model(self, model_name):
        """
        Train and evaluate the model, returning the best validation accuracy.
        """
        # Prepare the model and training arguments
        model = self.load_model(model_name)
        training_args = TrainingArguments(
            output_dir=f'../data/results/{model_name}',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_dir='../data/logs',
            load_best_model_at_end=True,
            seed=42,
        )

        # Fine-tune the model
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
        )

        start_time = time.time()
        print("\n*********************************************************\n")
        print(f'Starting training with model: {model_name}')
        trainer.train()
        print(f'Finished training with model: {model_name}')
        print('\n*********************************************************\n')
        training_time = time.time() - start_time
        return model, training_time


    def evaluate_model(self, model):
        """
        Evaluate the fine-tuned model on the validation set.
        :param model: Fine-tuned NER model.
        :return: Evaluation metrics (accuracy, f1-score, etc.).
        """
        print("\n\n*********************************************************\n\n")
        print(f'Starting evaluating with model: {model}')
        trainer = Trainer(model=model)
        start_time = time.time()
        predictions, labels, _ = trainer.predict(self.val_dataset)
        testing_time = time.time() - start_time
        predictions = predictions.argmax(axis=-1)
        accuracy = accuracy_score(labels.flatten(), predictions.flatten())
        return testing_time, accuracy

    def compare_models(self):
        """
        Compare multiple models based on performance metrics (accuracy, speed, etc.).
        :return: Best-performing model based on accuracy.
        """
        for model_name in self.models:
            print(f"Fine-tuning and evaluating {model_name}...")

            # Tokenize data
            self.tokenize_data(model_name)

            # Fine-tune the model
            model, training_time = self.train_model(model_name)

            # Evaluate the model
            testing_time, accuracy = self.evaluate_model(model)
            # accuracy = self.evaluate_model(model)
            self.model_performance[model_name] = {
                'accuracy': accuracy,
                'testing_time': testing_time,
                'training_time': training_time
            }
            print(f"Model: {model_name}, Accuracy: {accuracy} \n Training Time: {training_time} seconds, testing_time: {testing_time}")

        # Select the best model based on accuracy
        best_model_accuracy = max(self.model_performance, key=lambda x: self.model_performance[x]['accuracy'])
        best_model_speed = min(self.model_performance, key=lambda x: self.model_performance[x]['testing_time'])
        best_model_training_speed = min(self.model_performance, key=lambda x: self.model_performance[x]['training_time'])

        # Print best models in terms of already selected performance metrics
        print(f"Best model (Accuracy): {best_model_accuracy} with Accuracy: {self.model_performance[best_model_accuracy]['accuracy']}")
        print(f"Best model (Speed): {best_model_speed} with Speed: {self.model_performance[best_model_speed]['testing_time']}")
        print(f"Best model (Training Time): {best_model_training_speed} with Training Time: {self.model_performance[best_model_training_speed]['training_time']}")

        return best_model_accuracy, best_model_speed, best_model_training_speed
