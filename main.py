import sys

from src.components.data_ingestion import DataIngestion
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.config_entity import TrainingPipelineConfig
from src.components.data_validation import DataValidation

if __name__ == '__main__':
    try:
        training_config = TrainingPipelineConfig()

        # =========================================
        # Data ingestion
        # =========================================
        di_config = DataIngestionConfig(training_config)
        data_ingestion = DataIngestion(di_config)
        logging.info('Initiate data ingestion.')
        di_artifact = data_ingestion.initiate_data_ingestion()
        logging.info('Data ingestion completed')
        print(di_artifact)

        # =========================================
        # Data validation
        # =========================================
        dv_config = DataValidationConfig(training_config)
        data_validation = DataValidation(di_artifact, dv_config)
        logging.info('Initiate data validation')
        dv_artifact = data_validation.initiate_data_validation()
        logging.info('Data validation completed')
        print(dv_artifact)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
