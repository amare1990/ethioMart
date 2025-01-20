"""Importing necessaries libraries. """
import pandas as pd

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, pipeline



class NERModel:
  def __init__(self, model_name='XLM-Roberta', dataset_path='../data/labeled_dataset.conll'):
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

      # Process CoNLL formatted data into DataFrame
      data = []
      sentence = []
      labels = []

      for line in lines:
          if line.strip():  # If line is not empty
              token, label = line.strip().split()
              sentence.append(token)
              labels.append(label)
          else:
              # Save sentence with its labels
              data.append({'tokens': sentence, 'labels': labels})
              sentence, labels = [], []  # Reset for next sentence

      df = pd.DataFrame(data)
      df['labels'] = df['labels'].apply(lambda x: [self.map_labels(label) for label in x])

      # Split the dataset into train and validation sets
      train_df, val_df = train_test_split(df, test_size=0.2)

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

        def tokenize_and_align_labels(example):
            """
            Tokenize text and align the labels with the tokenized text.
            """
            # Tokenize text
            tokenized_inputs = self.tokenizer(example['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
            labels = example['labels']

            # Align labels with tokenized inputs
            word_ids = tokenized_inputs.word_ids()
            aligned_labels = []
            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)  # Ignore padding tokens
                else:
                    aligned_labels.append(labels[word_id])

            tokenized_inputs['labels'] = aligned_labels
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
            output_dir='./results',          # output directory
            num_train_epochs=3,              # number of training epochs
            per_device_train_batch_size=8,   # batch size for training
            per_device_eval_batch_size=16,   # batch size for evaluation
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
            logging_dir='./logs',            # directory for storing logs
            evaluation_strategy="epoch",     # evaluation strategy to use
            save_strategy="epoch",           # save model checkpoint after each epoch
            load_best_model_at_end=True,     # load the best model when finished training
        )
