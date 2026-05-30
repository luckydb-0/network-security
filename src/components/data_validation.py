from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.constants.training_pipeline import SCHEMA_FILE_PATH, TARGET_COLUMN
from src.utils.common.utils import read_yaml_file, write_yaml_file

from scipy.stats import ks_2samp
import pandas as pd
from pandas.api.types import is_float_dtype
import os
import sys

class DataValidation:
    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        try:
            n_columns = len(self._schema_config['columns'])
            logging.info(f'Required number of columns: {n_columns}')
            logging.info(f'Data frame has columns: {len(df.columns)}')
            return len(df.columns) == n_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validate_column_names(self, df: pd.DataFrame) -> bool:
        try:
            expected_columns = [
                column
                for column_config in self._schema_config['columns']
                for column in column_config
            ]
            missing_columns = [
                column for column in expected_columns if column not in df.columns
            ]
            unexpected_columns = [
                column for column in df.columns if column not in expected_columns
            ]

            logging.info(f'Missing columns: {missing_columns}')
            logging.info(f'Unexpected columns: {unexpected_columns}')
            return not missing_columns and not unexpected_columns
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validate_column_data_types(self, df: pd.DataFrame) -> bool:
        try:
            expected_data_types = {
                column: data_type
                for column_config in self._schema_config['columns']
                for column, data_type in column_config.items()
            }

            def has_expected_data_type(column: str, expected_data_type: str) -> bool:
                series = df[column]
                if str(series.dtype) == expected_data_type:
                    return True

                # Pandas promotes integer columns to float when they contain NaN.
                return (
                    expected_data_type.startswith('int')
                    and series.isna().any()
                    and is_float_dtype(series.dtype)
                    and (series.dropna() % 1 == 0).all()
                )

            invalid_data_types = {
                column: {
                    'expected': expected_data_type,
                    'actual': str(df[column].dtype)
                }
                for column, expected_data_type in expected_data_types.items()
                if column in df.columns
                and not has_expected_data_type(column, expected_data_type)
            }

            logging.info(f'Columns with invalid data types: {invalid_data_types}')
            return not invalid_data_types
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def validate_dataframe_not_empty(df: pd.DataFrame) -> bool:
        return not df.empty

    @staticmethod
    def validate_target_has_no_missing_values(df: pd.DataFrame) -> bool:
        return TARGET_COLUMN in df.columns and not df[TARGET_COLUMN].isna().any()

    @staticmethod
    def validate_features_not_entirely_missing(df: pd.DataFrame) -> bool:
        input_features = df.drop(columns=[TARGET_COLUMN], errors='ignore')
        entirely_missing_features = input_features.columns[input_features.isna().all()].tolist()
        logging.info(f'Entirely missing input features: {entirely_missing_features}')
        return not entirely_missing_features

    def validate_duplicate_rows(self, df: pd.DataFrame) -> bool:
        try:
            duplicate_row_ratio = df.duplicated().mean()
            max_duplicate_row_ratio = self._schema_config['max_duplicate_row_ratio']
            logging.info(
                f'Duplicate row ratio: {duplicate_row_ratio:.4f}; '
                f'maximum allowed: {max_duplicate_row_ratio:.4f}'
            )
            return duplicate_row_ratio <= max_duplicate_row_ratio
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    @staticmethod
    def validate_target_column(df: pd.DataFrame) -> bool:
        return TARGET_COLUMN in df.columns

    def validate_allowed_values(self, df: pd.DataFrame) -> bool:
        try:
            invalid_values = {}
            for column, allowed_values in self._schema_config['allowed_values'].items():
                if column not in df.columns:
                    continue
                unexpected_values = set(df[column].dropna().unique()) - set(allowed_values)
                if unexpected_values:
                    invalid_values[column] = sorted(unexpected_values)

            logging.info(f'Columns with invalid values: {invalid_values}')
            return not invalid_values
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def validate_dataframe(self, df: pd.DataFrame, dataframe_name: str) -> bool:
        try:
            checks = {
                'is not empty': self.validate_dataframe_not_empty(df),
                'has the required number of columns': self.validate_number_of_columns(df),
                'has the expected column names': self.validate_column_names(df),
                'has the expected data types': self.validate_column_data_types(df),
                f'does not contain missing values in target column {TARGET_COLUMN}': self.validate_target_has_no_missing_values(df),
                'does not contain entirely missing input features': self.validate_features_not_entirely_missing(df),
                'does not exceed the duplicate row limit': self.validate_duplicate_rows(df),
                f'contains target column {TARGET_COLUMN}': self.validate_target_column(df),
                'contains only allowed values': self.validate_allowed_values(df),
            }
            failed_checks = [check for check, passed in checks.items() if not passed]
            if failed_checks:
                logging.error(f'{dataframe_name} dataframe failed checks: {failed_checks}')
            return not failed_checks
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold = 0.05) -> bool:
        try:
            report = {}
            validation_status = True
            for col in base_df.columns:
                d1 = base_df[col].dropna()
                d2 = current_df[col].dropna()
                is_same_dist = ks_2samp(d1, d2) # Compare distribution of two samples
                drift_status = bool(is_same_dist.pvalue < threshold)
                if drift_status:
                    validation_status = False
                report.update({col: {
                        'p_value': float(is_same_dist.pvalue),
                        'drift_status': drift_status
                    }
                })

            drift_report_file_path = self.data_validation_config.drift_report_file_path

            # Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok = True)
            write_yaml_file(file_path = drift_report_file_path, content = report)
            return validation_status
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_filepath
            test_file_path = self.data_ingestion_artifact.test_filepath

            # Read data
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            train_status = self.validate_dataframe(train_df, 'Train')
            test_status = self.validate_dataframe(test_df, 'Test')
            quality_status = train_status and test_status

            # Exercise simplification: this checks split consistency. A production
            # drift check should compare incoming data with an approved baseline.
            drift_validation_status = (
                self.detect_dataset_drift(train_df, test_df)
                if quality_status
                else False
            )
            status = quality_status and drift_validation_status

            if not quality_status:
                logging.warning('Dataset drift check skipped because data quality validation failed.')

            if status:
                train_output_path = self.data_validation_config.valid_train_file_path
                test_output_path = self.data_validation_config.valid_test_file_path
            else:
                train_output_path = self.data_validation_config.invalid_train_file_path
                test_output_path = self.data_validation_config.invalid_test_file_path

            dir_path = os.path.dirname(train_output_path)
            os.makedirs(dir_path, exist_ok = True)

            train_df.to_csv(
                train_output_path, index = False, header = True
            )

            test_df.to_csv(
                test_output_path, index = False, header = True
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = train_output_path if status else None,
                valid_test_file_path = test_output_path if status else None,
                invalid_train_file_path = train_output_path if not status else None,
                invalid_test_file_path = test_output_path if not status else None,
                drift_report_file_path = (
                    self.data_validation_config.drift_report_file_path
                    if quality_status
                    else None
                ),
            )
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
