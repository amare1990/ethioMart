import os
BASE_DIR = os.path.join(__file__, '.env')
from telethon import TelegramClient

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

    def fetch_data_from_channels(self, channel_names):
        """
        Fetch data (messages, images, documents) from specified Telegram channels in real-time.
        """
        for channel_name in channel_names:
            channel = self.client.get_entity(channel_name)
            messages = self.client.iter_messages(channel)
            for msg in messages:
                if msg.text:
                    self.data.append({'message': msg.text, 'timestamp': msg.date, 'sender': msg.sender_id})
                if msg.media:
                    # Handling media (images, documents)
                    media = self.download_media(msg.media)
                    self.data.append({'media': media, 'timestamp': msg.date, 'sender': msg.sender_id})
