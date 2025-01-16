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
