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
        self.text_folder = os.path.join(self.main_download_folder, 'text_folder')
        self.media_folder = os.path.join(self.main_download_folder, 'media_folder')

        # Create main folder and subfolders if they don't exist
        os.makedirs(self.main_download_folder, exist_ok=True)
        os.makedirs(self.text_folder, exist_ok=True)
        os.makedirs(self.media_folder, exist_ok=True)

    async def download_media(self, media):
        """
        Download and process media (images, documents) shared in the messages.
        """
        if media:
            # Specify media_folder to download the media files
            # media_folder = os.path.join(self.download_folder, 'media_folder')
            media_folder = os.path.join(self.main_download_folder, 'media_folder')
            if not os.path.exists(media_folder):
                os.makedirs(media_folder)

            # Download and save the media to the media_folder
            media_path = await self.client.download_media(media, file=self.media_folder)
            print(f"Media saved to {media_path}")  # Optionally print the saved media path
            return media_path
        return None


    def save_text_data(self, text, filename):
        """
        Save text data in the text folder.
        """
        text_path = os.path.join(self.text_folder, f'{filename}.txt')
        with open(text_path, 'w', encoding='utf-8') as file:
            file.write(text)
        return text_path

    def compress_downloaded_files(self, zip_name='downloaded_files.zip'):
        """
        Compress the downloaded text and media files into a ZIP archive.
        """
        # Compress both text_folder and media_folder into one zip file
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
        for channel_name in channel_names:
            try:
                channel = await self.client.get_entity(channel_name)
                async for msg in self.client.iter_messages(channel):
                    if msg.text:
                        # Save text data to text folder
                        self.save_text_data(msg.text, f'text_{msg.date.timestamp()}')
                        self.data.append({
                            'type': 'text',
                            'content': msg.text,
                            'timestamp': msg.date,
                            'sender': msg.sender_id
                        })
                    if msg.media:
                        # Handle media if necessary (e.g., download)
                        media_path = await self.download_media(msg.media)
                        self.data.append({
                            'type': 'media',
                            # 'content': str(msg.media),  # Placeholder for media
                            'content': media_path,  # Store the media path
                            'timestamp': msg.date,
                            'sender': msg.sender_id
                        })
            except Exception as e:
                print(f"Error fetching data from {channel_name}: {e}")
        print("Data fetched successfully!")

    def preprocess_text(self, text):
        """
        Preprocess Amharic text: tokenization, normalization, and handling language-specific features.
        """
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
                cleaned_data.append({
                    'message': self.preprocess_text(entry['content']),
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
        df.to_csv(f'../data/{filename}', index=False)
        print(f"Data stored in {filename}")
