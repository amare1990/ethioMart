import os
import re
# BASE_DIR = os.path.join(__file__, '.env')
from telethon import TelegramClient
from PIL import Image
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()



class DataProcessor:
    def __init__(self, phone_number):
        # Telegram client initialization

        self.api_id = os.getenv('API_ID')
        self.api_hash = os.getenv('API_HASH')
        self.client = TelegramClient(phone_number, self.api_id, self.api_hash)
        self.channels = []  # List to store selected channels
        self.data = []  # List to store preprocessed data

    def connect_to_telegram(self):
        """
        Connect to Telegram using the Telethon client.
        """
        self.client.start()
        print("Connected to Telegram")

    async def fetch_data_from_channels(self, channel_names):
        """
        Fetch data (messages, images, documents) from specified Telegram channels.
        """
        for channel_name in channel_names:
            channel = await self.client.get_entity(channel_name)  # Await entity retrieval
            async for msg in self.client.iter_messages(channel):
                if msg.text:
                    self.data.append({
                        'type': 'text',
                        'content': msg.text,
                        'timestamp': msg.date,
                        'sender': msg.sender_id
                    })
                if msg.media:
                    # Handling media (images, documents, etc.)
                    media_path = await self.download_media(msg.media)
                    self.data.append({
                        'type': 'media',
                        'content': media_path,
                        'timestamp': msg.date,
                        'sender': msg.sender_id
                    })
        print("Data with media fetched successfully")

    async def download_media(self, media):
        """
        Download and process media (images, documents) shared in the messages.
        """
        # Check if the media is valid and can be downloaded
        if media:
            # Use Telethon's built-in `download_media` method
            media_path = await self.client.download_media(media)
            return media_path
        return None

    def preprocess_text(self, text):
        """
        Preprocess Amharic text: tokenization, normalization, and handling language-specific features.
        """
        # Tokenization example (simple whitespace-based tokenization for Amharic)
        tokens = text.split()

        # Normalize text (removing extra spaces, punctuation, etc.)
        text = ' '.join(tokens)
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation

        # Add Amharic-specific preprocessing steps if needed (e.g., handling specific characters)
        # Example: Normalize or remove common non-alphabetic characters

        return text

    def clean_and_structure_data(self):
        """
        Clean and structure the data into a unified format.
        """
        cleaned_data = []
        for entry in self.data:
            if 'message' in entry:
                cleaned_data.append({
                    'message': self.preprocess_text(entry['message']),
                    'sender': entry['sender'],
                    'timestamp': entry['timestamp']
                })
            if 'media' in entry:
                cleaned_data.append({
                    'media': entry['media'],
                    'sender': entry['sender'],
                    'timestamp': entry['timestamp']
                })
        self.data = cleaned_data

