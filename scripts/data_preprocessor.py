from telethon import TelegramClient



class DataProcessor:
    def __init__(self, api_id, api_hash, phone_number):
        # Telegram client initialization
        self.client = TelegramClient(phone_number, api_id, api_hash)
        self.channels = []  # List to store selected channels
        self.data = []  # List to store preprocessed data

    def connect_to_telegram(self):
        """
        Connect to Telegram using the Telethon client.
        """
        self.client.start()
        print("Connected to Telegram")
