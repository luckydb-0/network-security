import sys

from src.components.data_ingestion import DataIngestion
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import TrainingPipelineConfig

if __name__ == '__main__':
    try:
        training_config = TrainingPipelineConfig()
        dc_config = DataIngestionConfig(training_config)
        data_ingestion = DataIngestion(dc_config)
        logging.info('Initiate data ingestion.')
        artifact = data_ingestion.initiate_data_ingestion()
        print(artifact)
    except Exception as e:
        raise NetworkSecurityException(e, sys)