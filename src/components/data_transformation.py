import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from src.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.utils.common.utils import save_numpy_array_data, save_object


class DataTransformation:
    def __init__(self,
                 data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def get_data_transformer_object(self) -> Pipeline:
        """
        It initialises a KNNImputer object with the parameters specified in the training_pipeline.py file
        and returns a Pipeline object with the KNNImputer object as the first step.

        Args:
            cls: DataTransformation

        Returns:
            A Pipeline object
        """
        logging.info('Entered get_data_transformer_object method of DataTransformation class')
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info(f'Initialising KNNImputer with {DATA_TRANSFORMATION_IMPUTER_PARAMS}')
            return Pipeline([('imputer', imputer)])
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info('Starting data transformation...')
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
        
            train_input_features = train_df.drop(columns=[TARGET_COLUMN])
            train_target_feature = train_df[TARGET_COLUMN]
            train_target_feature = train_target_feature.replace(-1, 0)
        
            test_input_features = test_df.drop(columns=[TARGET_COLUMN])
            test_target_feature = test_df[TARGET_COLUMN]
            test_target_feature = test_target_feature.replace(-1, 0)

            pipeline = self.get_data_transformer_object()
            preprocess_obj = pipeline.fit(train_input_features)
            train_input_features_transformed = preprocess_obj.transform(train_input_features)
            test_input_features_transformed = preprocess_obj.transform(test_input_features)

            train_array = np.c_[train_input_features_transformed, np.array(train_target_feature)]
            test_array = np.c_[test_input_features_transformed, np.array(test_target_feature)]

            # Save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array = train_array)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array = test_array)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocess_obj)

            # Prepare artifacts
            return DataTransformationArtifact(
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e