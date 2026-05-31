import sys

from src.components.data_ingestion import DataIngestion
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from src.entity.config_entity import TrainingPipelineConfig
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation


def run_pipeline():
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

    if not dv_artifact.validation_status:
        logging.warning('Data validation failed. Skipping data transformation.')
        return dv_artifact

    # =========================================
    # Data transformation
    # =========================================
    df_config = DataTransformationConfig(training_config)
    data_transformation = DataTransformation(dv_artifact, df_config)
    logging.info('Initiate data transformation')
    df_artifact = data_transformation.initiate_data_transformation()
    logging.info('Data transformation completed')
    print(df_artifact)
    return df_artifact


if __name__ == '__main__':
    try:
        run_pipeline()
    except Exception as e:
        raise NetworkSecurityException(e, sys)
