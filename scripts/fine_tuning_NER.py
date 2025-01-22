# Fine tuning NER using pre-trained models
"""Importing necessaries libraries. """
import pandas as pd

from sklearn.model_selection import train_test_split
from datasets import Dataset

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments



class NERModel:
  def __init__(self, model_name='XLM-Roberta-base', dataset_path = '../data/labeled_dataset.conll'):
    """
    Initialize the NER model class with the model name and dataset path.
    :param model_name: The pre-trained model to use for fine-tuning (default: 'xlm-roberta-base').
    :param dataset_path: Path to the labeled dataset in CoNLL format (default: 'labeled_data.conll').
    """
    self.model_name = model_name
    self.dataset_path = dataset_path
    self.model = None
    self.tokenizer = None
    self.train_dataset = None
    self.val_dataset = None
    self.trainer = None

  def load_data(self):
        """
        Load the labeled dataset (in CoNLL format) and prepare the data for training.
        """
        # Load CoNLL data and convert it to a pandas DataFrame
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Process CoNLL formatted data into a list of dictionaries
        data = []
        sentence = []
        labels = []

        for line in lines:
            if line.strip():  # If line is not empty
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    token, label = parts
                    sentence.append(token)
                    labels.append(label)
            else:
                if sentence and labels:  # Ensure there is a valid sentence and labels
                    data.append({'tokens': sentence, 'labels': labels})
                    sentence, labels = [], []  # Reset for the next sentence

        # Handle any remaining sentence at the end of the file
        if sentence and labels:
            data.append({'tokens': sentence, 'labels': labels})

        # Convert list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        # Map labels to their corresponding IDs
        df['labels'] = df['labels'].apply(lambda x: [self.map_labels(label) for label in x])

        # Split the dataset into train and validation sets
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Convert DataFrames to Hugging Face Dataset format
        self.train_dataset = Dataset.from_pandas(train_df)
        self.val_dataset = Dataset.from_pandas(val_df)

  def map_labels(self, label):
        """
        Map CoNLL labels to a numerical value for token classification.
        :param label: CoNLL entity label.
        :return: Numerical label
        """
        label_map = {
            'B-Product': 0, 'I-Product': 1,
            'B-LOC': 2, 'I-LOC': 3,
            'B-PRICE': 4, 'I-PRICE': 5,
            'O': 6
        }
        return label_map.get(label, 6)  # Default to 'O' for outside entities

  def tokenize_data(self):
      """
      Tokenize the data and align the labels with the tokens.
      """
      # Load pre-trained tokenizer
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

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
                      label_ids.append(labels[word_idx])
                  else:
                      label_ids.append(-100)
                  previous_word_idx = word_idx
              all_ner_tags.append(label_ids)

          tokenized_inputs['labels'] = all_ner_tags
          return tokenized_inputs

      # Apply tokenization and label alignment
      self.train_dataset = self.train_dataset.map(tokenize_and_align_labels, batched=True)
      self.val_dataset = self.val_dataset.map(tokenize_and_align_labels, batched=True)

  def load_model(self):
        """
        Load the pre-trained model for token classification (NER).
        """
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, num_labels=7)
  def train_model(self):
        """
        Fine-tune the NER model using Hugging Face Trainer API.
        """
        training_args = TrainingArguments(
            output_dir='../data/fine_tuning/results',          # output directory
            num_train_epochs=10,              # number of training epochs
            per_device_train_batch_size=8,   # batch size for training
            per_device_eval_batch_size=16,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='../data/fine_tuning/logs',            # directory for storing logs
            evaluation_strategy="epoch",     # evaluation strategy to use
            save_strategy="epoch",           # save model checkpoint after each epoch
            load_best_model_at_end=True,     # load the best model when finished training
            seed=42,                         # random seed for reproducibility
        )

        self.trainer = Trainer(
          model=self.model,                         # the model to train
          args=training_args,                       # training arguments
          train_dataset=self.train_dataset,         # training dataset
          eval_dataset=self.val_dataset,            # validation dataset
        )

         # Train the model
        self.trainer.train()

  def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the validation set.
        """
        results = self.trainer.evaluate()
        print(f"Validation Results: {results}")
        return results

  def save_model(self, output_dir='../data/fine_tuning/fine_tuned_model'):
        """
        Save the fine-tuned model for future use.
        """
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved in {output_dir}")

