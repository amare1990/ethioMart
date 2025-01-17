import os
import re



class CoNLLLabeler:
    def __init__(self, dataset):
        """
        Initialize with the provided dataset.
        :param dataset: Pandas DataFrame containing the messages.
        """
        self.data = dataset
        self.labeled_data = []
