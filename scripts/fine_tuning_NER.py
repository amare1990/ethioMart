"""Importing necessaries libraries. """
import pandas as pd



class NERModel:
  def __init__(self, model_name='XLM-Roberta', data_path='../data/labeled_dataset.conll'):
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
