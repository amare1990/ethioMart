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

        location_keywords = ['Addis Ababa', 'ለቡ', 'ለቡ መዳህኒዓለም', 'መገናኛ', 'ቦሌ', 'ሜክሲኮ', "ፒያሳ", "ጊዮርጊስ",
                             "ተፋር", "ራመት", "ታቦር", "ኦዳ", "ሱቅ", "ቁ1መገናኛ", "ቁ2ፒያሳ", "4 ኪሎ", "4ኪሎ", "ቃሊቲ", "ሰሚት",
                             "summit", "ሲኤምሲ", "CMC", "ሐያት", "ሀያት", "hayat", "ካሳንችስ", "ካሣንችስ", "ካዛንችስ",
                             "ሃያ ሁለት", "ሃያሁለት", "ሾላ", "መርካቶ", "ሜክሲኮ", "Mexico", "mexico", "Mercato", "mercato",
                             "merkato", "ድሬ ዳዋ", "ባሕር ዳር", "ባህር ዳር", "ጎንደር", "ደሴ", "ሃዋሳ", "ሐዋሳ", "አዋሳ", "አዳማ",
                              "ሐረር", "መቀሌ"
        ]

        # Process each line for entity recognition
        for line in lines:
            tokens = re.findall(r'\S+', line)  # Tokenize the line

            # Process product entities
            if len(tokens) >= 5:  # Example threshold for B-PRODUCT
                product_name = tokens[:5]  # First 5 tokens are part of B-PRODUCT
                labeled_tokens.append(f"{product_name[0]} B-PRODUCT")
                for token in product_name[1:]:
                    labeled_tokens.append(f"{token} I-PRODUCT")

            # Process price entities
            for i, token in enumerate(tokens):
                # Check for price patterns with "ብር" included
                bundled_match = re.match(r'(\d+)(ብር)', token)  # Match "400ብር"
                if bundled_match:
                    # Split and label the price and currency
                    price, currency = bundled_match.groups()
                    labeled_tokens.append(f"ዋጋ B-PRICE") if token == "ዋጋ" else None
                    labeled_tokens.append(f"{price} I-PRICE")
                    labeled_tokens.append(f"{currency} O")
                    break

                # Check for "ዋጋ" preceding the price
                if token == "ዋጋ" and i + 1 < len(tokens):
                    labeled_tokens.append(f"{token} B-PRICE")
                    # Check the next token for price and "ብር" together or separated
                    next_token = tokens[i + 1]
                    if re.match(r'\d+\s*ብር', next_token):  # Match price with "ብር"
                        price_match = re.match(r'(\d+)', next_token)  # Extract just the number
                        if price_match:
                            labeled_tokens.append(f"{price_match.group(1)} I-PRICE")
                    elif re.match(r'\d+', next_token) and i + 2 < len(tokens) and tokens[i + 2] == "ብር":
                        # Handle separated price and "ብር" (e.g., "800 ብር")
                        labeled_tokens.append(f"{next_token} I-PRICE")
                        labeled_tokens.append(f"{tokens[i + 2]} O")  # Label "ብር" as outside
                    break

                # Directly label the token if it matches the price pattern
                if re.match(r'\d+\s*ብር', token) and not 'ዋጋ' in token:
                    if token not in [t.split()[0] for t in labeled_tokens]:  # Check if token is already labeled
                        labeled_tokens.append(f"{token} B-PRICE")
                        price_match = re.match(r'(\d+)', token)  # Extract only the number
                        if price_match:
                            labeled_tokens.append(f"{price_match.group(1)} I-PRICE")
                    break

                # Handle "ብር" separately if not part of a price pattern
                if token == "ብር":
                    labeled_tokens.append(f"{token} O")
                else:
                    labeled_tokens.append(f"{token} O")

            # Process location entities
            if "አድራሻ" in tokens:
                for i, token in enumerate(tokens):
                    if token == "አድራሻ" and i + 1 < len(tokens):  # Ensure there is a next token
                        # Label the next token as B-LOC
                        labeled_tokens.append(f"{tokens[i + 1]} B-LOC")

                        # Label the next eight tokens but skip one in between
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
