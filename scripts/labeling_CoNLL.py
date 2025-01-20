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
        # Split the message into lines
        lines = message.split('\n')
        labeled_tokens = []

        # Define keywords for B-LOC, I-LOC
        location_keywords = ['Addis Ababa', 'ለቡ', 'ለቡ መዳህኒዓለም', 'መገናኛ', 'ቦሌ', 'ሜክሲኮ', "ፒያሳ", "ጊዮርጊስ", "ተፋር", "ራመት", "ታቦር", "ኦዳ", "ሱቅ"]

        # Process each line for entity recognition
        for line in lines:
            tokens = re.findall(r'\S+', line)  # Tokenize the line


            # Process product entities
            if len(tokens) >= 5:  # Example threshold for B-PRODUCT
                product_name = tokens[:5]  # First 5 tokens are part of B-PRODUCT
                product = ' '.join(product_name)
                labeled_tokens.append(f"{product} B-PRODUCT")
                for token in product_name[1:]:
                    labeled_tokens.append(f"{token} I-PRODUCT")

            # Process price entities
            for i, token in enumerate(tokens):
                # Match prices like "ዋጋ 400ብር" where ዋጋ is the label and 400ብር is the price
                if re.match(r'ዋጋ \d+ብር', ' '.join(tokens[i:i+2])):  # Match the "ዋጋ <number>ብር" pattern
                    b_price = ' '.join(tokens[i:i+2])
                    labeled_tokens.append(f"{b_price} B-PRICE")  # Label "ዋጋ 400ብር" as B-PRICE

                    # Extract the number part (e.g., 400) and label it as I-PRICE
                    number_part = re.search(r'\d+', tokens[i + 1]).group()  # Extract number from the second token
                    labeled_tokens.append(f"{number_part} I-PRICE")  # Label the number part as I-PRICE

            # Process location entities
            for i, token in enumerate(tokens):
                if any(loc in token for loc in location_keywords):
                    if i == 0 or tokens[i - 1] not in ["ቁ", "ቦሌ"]:  # Avoid false positives
                        labeled_tokens.append(f"{token} B-LOC")
                    else:
                        labeled_tokens.append(f"{token} I-LOC")

            # Default to O for other tokens (outside any entities)
            for token in tokens:
                if token not in [t.split()[0] for t in labeled_tokens]:  # Check if token is already labeled
                    labeled_tokens.append(f"{token} O")

        return "\n".join(labeled_tokens)




    def label_dataset(self, max_messages=30):
        """
        Process the dataset and label each message.
        """
        for i, message in enumerate(self.data["message"]):
            if i >= max_messages:
                break  # Stop after labeling the specified number of messages

            # Label the message
            labeled_message = self.label_message_utf8_with_birr(message)

            # Append original message and labeled message to the dataset
            self.labeled_data.append("Original Message:")
            self.labeled_data.append(message)
            self.labeled_data.append("\nLabeled Message:")
            self.labeled_data.append(labeled_message)
            self.labeled_data.append("")  # Separate entries with a blank line

    def save_to_file(self, filename="../data/labeled_dataset.conll"):
        """
        Save the labeled data to a file in CoNLL format.
        :param filename: Name of the file to save the data.
        """
        with open(filename, "w", encoding="utf-8") as file:
            file.write("\n".join(self.labeled_data))
