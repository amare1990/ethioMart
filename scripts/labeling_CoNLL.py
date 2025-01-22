"""Rule-based labeling. """
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
      # Convert message to string if it's not already
        message = str(message)  # This line is added to handle float values
        # Split the message into lines
        lines = message.split('\n')
        labeled_tokens = []

        # Define keywords for B-LOC, I-LOC

        location_keywords = ['Addis Ababa', 'ለቡ', 'ለቡ መዳህኒዓለም', 'መገናኛ', 'ቦሌ', 'ሜክሲኮ', "ፒያሳ", "ጊዮርጊስ",
                             "ተፋር", "ራመት", "ታቦር", "ኦዳ", "ሱቅ", "ቁ1መገናኛ", "ቁ2ፒያሳ", "4 ኪሎ", "4ኪሎ", "ቃሊቲ", "ሰሚት",
                             "summit", "ሲኤምሲ", "CMC", "ሐያት", "ሀያት", "hayat", "ካሳንችስ", "ካሣንችስ", "ካዛንችስ",
                             "ሃያ ሁለት", "ሃያሁለት", "ሾላ", "መርካቶ", "ሜክሲኮ", "Mexico", "mexico", "Mercato", "mercato", "merkato"
        ]

        # Process each line for entity recognition
        for line in lines:
            tokens = re.findall(r'\S+', line)  # Tokenize the line

            # Process product entities
            # List of invitation calls or phrases to exclude from being labeled as products
            exclusion_phrases = [
                "ረፍት ቀንዎ",
                "ደንበኞቻችን",
                "ሱቅ ላይ መስተናገድ",
                "በረፍት ቀንዎ",
                "ምትፈልጉ ውድ"
                # Add any other exclusion phrases here
            ]
            if any(phrase in line for phrase in exclusion_phrases):
                # Skip product labeling if the phrase matches the exclusion list
                continue
            elif len(tokens) >= 5:  # Example threshold for B-PRODUCT
                product_name = tokens[:5]  # First 5 tokens are part of B-PRODUCT
                labeled_tokens.append(f"{product_name[0]} B-PRODUCT")
                for token in product_name[1:]:
                    labeled_tokens.append(f"{token} I-PRODUCT")

            # Process price entities
            for i, token in enumerate(tokens):
                # Check for price patterns with "ብር" included in the same token (e.g., "400ብር")
                bundled_match = re.match(r'(\d+)(ብር)', token)  # Match "400ብር"
                if bundled_match:
                    # Split and label the price and currency
                    price, currency = bundled_match.groups()
                    labeled_tokens.append(f"{token} B-PRICE") if token == "ዋጋ" else None
                    labeled_tokens.append(f"{price} I-PRICE")  # Only the numeric part is I-PRICE
                    labeled_tokens.append(f"{currency} O")  # Currency "ብር" is labeled as O
                    break

                # Check for "ዋጋ" preceding the price
                if token == "ዋጋ" and i + 1 < len(tokens):
                    labeled_tokens.append(f"{token} B-PRICE")
                    # Process tokens following "ዋጋ"
                    j = i + 1
                    while j < len(tokens):
                        next_token = tokens[j]
                        # Handle prices like "400", "400ብር", or alphanumeric tokens like "22L"
                        if re.match(r'\d+[A-Za-z]*', next_token):  # Match numeric or alphanumeric price
                            bundled_match = re.match(r'(\d+)(ብር)?', next_token)  # Match "400ብር" or "400"
                            if bundled_match:
                                price, currency = bundled_match.groups()
                                labeled_tokens.append(f"{price} I-PRICE")  # Label the numeric price
                                if currency:  # If "ብር" is present, label it as O
                                    labeled_tokens.append(f"{currency} O")
                        else:
                            break
                        j += 1
                    break



            # Process location entities
            if "አድራሻ" in tokens:
                for i, token in enumerate(tokens):
                    if token == "አድራሻ" and i + 1 < len(tokens):  # Ensure there is a next token
                        # Label the next token as B-LOC
                        labeled_tokens.append(f"{tokens[i + 1]} B-LOC")

                        # Label the next five tokens but skip one in between
                        for j in range(2, 10):  # From i+2 to i+6 (5 tokens)
                            if i + j < len(tokens):  # Ensure we don't go out of bounds
                                labeled_tokens.append(f"{tokens[i + j]} I-LOC")
            else:
                # Handle the case where "አድራሻ" is not in the tokens
                for i, token in enumerate(tokens):
                    # Match token against location_tokens using regex
                    if any(re.fullmatch(loc, token) for loc in location_keywords):
                        # Label the first matching token as B-LOC
                        labeled_tokens.append(f"{token} B-LOC")

                        # Label the next nine tokens as I-LOC
                        for j in range(1, 10):  # Up to the next three tokens
                            if i + j < len(tokens):
                                labeled_tokens.append(f"{tokens[i + j]} I-LOC")
                        break  # Exit loop after labeling the first match

            # # Print the labeled tokens
            # print(labeled_tokens)


            # Default to O for other tokens (outside any entities)
            for token in tokens:
                if token not in [t.split()[0] for t in labeled_tokens] and not re.match(r'(\d)ብር', token):

                    labeled_tokens.append(f"{token} O")

        return "\n".join(labeled_tokens)



    def label_dataset(self):
        """
        Process the dataset and label each message.
        """
        for message in self.data["message"]:
            # if i >= max_messages:
            #     break  # Stop after labeling the specified number of messages

            # Label the message
            labeled_message = self.label_message_utf8_with_birr(message)

            # Append original message and labeled message to the dataset
            # self.labeled_data.append("Original Message:")
            # self.labeled_data.append(message)
            # self.labeled_data.append("\nLabeled Message:")
            self.labeled_data.append(labeled_message)
            self.labeled_data.append("")  # Separate entries with a blank line

    def save_to_file(self, filename="../data/labeled_dataset.conll"):
        """
        Save the labeled data to a file in CoNLL format.
        :param filename: Name of the file to save the data.
        """
        with open(filename, "w", encoding="utf-8") as file:
            file.write("\n".join(self.labeled_data))
