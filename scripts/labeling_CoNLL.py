import re



class CoNLLLabeler:
    def __init__(self, dataset):
        """
        Initialize with the provided dataset.
        :param dataset: Pandas DataFrame containing the messages.
        """
        self.data = dataset
        self.labeled_data = []

    def label_message_utf8_with_birr(self, message):
        """
        Label a single message in CoNLL format, identifying entities like Product, Price, and Location.
        :param message: A string containing the message text.
        :return: Labeled message in CoNLL format.
        """
        labeled_tokens = []
        product_started = False

        # Split message by lines
        lines = message.split('\n')

        for line in lines:
            # Tokenize each line
            tokens = re.findall(r'\S+', line)

            for i, token in enumerate(tokens):
                if re.match(r'^\d+(\.\d{1,2})?$|ETB|ዋጋ|ብር|\$', token):
                    # Price entity
                    if i == 0 or not product_started:
                        labeled_tokens.append(f"{token} B-PRICE")
                    else:
                        labeled_tokens.append(f"{token} I-PRICE")
                elif token in ['Addis', 'Ababa', 'ለቡ', 'ቦሌ', 'ሜክሲኮ']:
                    # Location entity
                    if i == 0 or not product_started:
                        labeled_tokens.append(f"{token} B-LOC")
                    else:
                        labeled_tokens.append(f"{token} I-LOC")
                elif not product_started:
                    # First token of a product
                    labeled_tokens.append(f"{token} B-PRODUCT")
                    product_started = True
                else:
                    # Inside a product or general text
                    labeled_tokens.append(f"{token} I-PRODUCT")

            # Reset for the next line
            product_started = False

        return "\n".join(labeled_tokens)


    def label_dataset(self, max_messages=50):
        """
        Process the dataset and label each message.
        """
        for i, message in enumerate(self.data["Message"]):
            if i >= max_messages:
                break  # Stop after labeling the specified number of messages
            labeled_message = self.label_message_utf8_with_birr(message)
            self.labeled_data.append(labeled_message)
            self.labeled_data.append("")  # Separate messages with a blank line
