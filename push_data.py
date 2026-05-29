import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging

from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the MongoDB connection URL from the environment variables
MONGO_DB_URL = os.getenv('MONGO_DB_URL')

# Get the path to the trusted CA certificate bundle
# This can be useful when connecting securely to MongoDB Atlas
ca = certifi.where()


class NetworkDataExtract():
    """
    Class responsible for extracting network security data from a CSV file
    and inserting it into a MongoDB collection.
    """

    def __init__(self):
        """
        Initialize the NetworkDataExtract object.

        At the moment, this constructor does not initialize anything,
        but the try-except block is kept for consistent exception handling.
        """
        try:
            pass
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def csv_to_json_converter(self, file_path):
        """
        Convert a CSV file into a list of JSON-like Python dictionaries.

        Args:
            file_path: Path to the CSV file.

        Returns:
            A list of records, where each record is represented as a dictionary.
        """
        try:
            # Read the CSV file into a pandas DataFrame
            data = pd.read_csv(file_path)

            # Reset the DataFrame index and drop the old index
            data.reset_index(drop=True, inplace=True)

            # Convert the DataFrame into JSON format, then into Python dictionaries
            # data.T.to_json() creates a JSON object where each row becomes a record
            records = list(json.loads(data.T.to_json()).values())

            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def insert_data_mongodb(self, records, database, collection):
        """
        Insert records into a MongoDB collection.

        Args:
            records: List of dictionaries to insert.
            database: Name of the MongoDB database.
            collection: Name of the MongoDB collection.

        Returns:
            The number of records inserted into MongoDB.
        """
        try:
            # Store database name, collection name, and records as instance variables
            self.database = database
            self.collection = collection
            self.records = records

            # Create a MongoDB client using the connection URL
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            # Select the target database
            self.database = self.mongo_client[self.database]

            # Select the target collection inside the database
            self.collection = self.database[self.collection]

            # Insert all records into the MongoDB collection
            self.collection.insert_many(self.records)

            return len(self.records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)


if __name__ == '__main__':
    FILE_PATH = 'data/phishingData.csv'
    DATABASE = 'NETWORK-TEST'
    COLLECTION = 'NetworkData'

    netObj = NetworkDataExtract()
    records = netObj.csv_to_json_converter(file_path=FILE_PATH)

    # Insert the records into MongoDB
    no_of_records = netObj.insert_data_mongodb(records, DATABASE, COLLECTION)

    print(f'Number of records stored on MongoDB: {no_of_records}')