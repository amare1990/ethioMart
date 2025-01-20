"""Importing necessaries libraries. """
import pandas as pd

from sklearn.model_selection import train_test_split



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
