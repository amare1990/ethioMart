import os
import re

import numpy as np
import pandas as pd
# BASE_DIR = os.path.join(__file__, '.env')
from telethon import TelegramClient

from PIL import Image
from io import BytesIO

import shutil

from dotenv import load_dotenv
load_dotenv()

from telethon.sessions import MemorySession
import nest_asyncio

# Apply nested asyncio for Jupyter Notebooks
nest_asyncio.apply()


# List of Telegram channels to fetch data from (sample names)
channels = ['@ZemenExpress', '@nevacomputer', '@Shewabrand', '@modernshoppingcenter', '@aradabrand2']



class DataProcessor:
    def __init__(self, phone_number, download_folder='downloads'):
        # Telegram client initialization

        self.api_id = os.getenv('API_ID')
        self.api_hash = os.getenv('API_HASH')
        self.client = TelegramClient(MemorySession(), self.api_id, self.api_hash)
        self.channels = []  # List to store selected channels
        self.data = []  # List to store preprocessed data

        self.main_download_folder = download_folder
        os.makedirs(self.main_download_folder, exist_ok=True)  # Ensure folder exists

    async def download_media(self, media, filename):
        """
        Download and process media (images, documents) shared in the messages.
        """
        if media:
            media_path = os.path.join(self.main_download_folder, filename)
            media_path = await self.client.download_media(media, file=media_path)
            print(f"Media saved to {media_path}")
            return media_path
        return None

    def save_text_data(self, text, filename):
        """
        Save text data in the main download folder.
        """
        text_path = os.path.join(self.main_download_folder, f'{filename}.txt')
        with open(text_path, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text saved to {text_path}")
        return text_path

    def compress_downloaded_files(self, zip_name='downloaded_files.zip'):
        """
        Compress the downloaded files into a ZIP archive.
        """
        zip_path = os.path.join(self.main_download_folder, zip_name)
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', self.main_download_folder)
        print(f"Files compressed into {zip_path}")
        return zip_path

    async def connect_to_telegram(self):
        """
        Connect to Telegram using the Telethon client.
        """
        await self.client.start()
        print("Connected to Telegram")

    async def fetch_data_from_channels(self, channel_names):
        """
        Fetch data from the specified Telegram channels.
        """
        count_text = 0
        for channel_name in channel_names:
            try:
                channel = await self.client.get_entity(channel_name)
                async for msg in self.client.iter_messages(channel):
                    timestamp = int(msg.date.timestamp())
                    sender_id = msg.sender_id

                    if msg.text:
                        # Save text data with a timestamp-based filename
                        filename = f'text_{sender_id}_{timestamp}'
                        text_path = self.save_text_data(msg.text, filename)
                        self.data.append({
                            'type': 'text',
                            'content': text_path,
                            'timestamp': msg.date,
                            'sender': sender_id
                        })
                        print(f'Downloading from {channel_name}')
                        count_text =  count_text + 1
                        print(f'Downloaded the {++count_text}th text file')

                    # if msg.media:
                    #     count_media = 0
                    #     # Save media with a timestamp-based filename
                    #     filename = f'media_{sender_id}_{timestamp}'
                    #     media_path = await self.download_media(msg.media, filename)
                    #     self.data.append({
                    #         'type': 'media',
                    #         'content': media_path,
                    #         'timestamp': msg.date,
                    #         'sender': sender_id
                    #     })
                    #     print(f'Downloaded the {++count}th file')
                    #     print(f'Downloaded the {++count_media}th media file')
            except Exception as e:
                print(f"Error fetching data from {channel_name}: {e}")
        print("Data fetched successfully!")
    def remove_emojis(self, text):
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # geometric shapes extended
            "\U0001F800-\U0001F8FF"  # supplemental arrows-c
            "\U0001F900-\U0001F9FF"  # supplemental symbols & pictographs
            "\U0001FA00-\U0001FA6F"  # chess symbols
            "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-a
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub(r'', text)

    def preprocess_text(self, text):
        """
        Preprocess Amharic text: tokenization, normalization, and handling language-specific features.
        """
        # Remove emojis
        text = self.remove_emojis(text)

        # Tokenization example (simple whitespace-based tokenization for Amharic)
        tokens = text.split()

        # Normalize text (removing extra spaces, punctuation, etc.)
        text = ' '.join(tokens)
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

        return text

    def clean_and_structure_data(self):
        """
        Clean and structure the data into a unified format.
        """
        cleaned_data = []
        for entry in self.data:
            if entry['type'] == 'text':
                with open(entry['content'], 'r', encoding='utf-8') as file:
                    text = file.read()
                # Read the file content
                with open(entry['content'], 'r', encoding='utf-8') as file:
                    text_content = file.read()
                # Preprocess the text
                processed_text = self.preprocess_text(text_content)
                cleaned_data.append({
                    # 'message': self.preprocess_text(entry['content']),
                    'message': processed_text,  # Save the processed text instead of the file path
                    'sender': entry['sender'],
                    'timestamp': entry['timestamp']
                })
            if entry['type'] == 'media':
                cleaned_data.append({
                    'media': entry['content'],
                    'sender': entry['sender'],
                    'timestamp': entry['timestamp']
                })
        self.data = cleaned_data

    def store_data(self, filename='preprocessed_data.csv'):
        """
        Store the preprocessed data into a structured format (CSV for simplicity).
        """
        # Converting the structured data into a Pandas DataFrame
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"Data stored in {filename}")
