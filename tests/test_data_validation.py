import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
from sklearn.model_selection import train_test_split

from src.components.data_validation import DataValidation
from src.entity.artifact_entity import DataIngestionArtifact


class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.train_df, self.test_df = train_test_split(
            pd.read_csv('data/phishingData.csv'),
            test_size=0.2,
            random_state=1
        )
        ingested_dir = self.root / 'ingested'
        ingested_dir.mkdir()
        self.train_path = ingested_dir / 'train.csv'
        self.test_path = ingested_dir / 'test.csv'
        self.train_df.to_csv(self.train_path, index=False)
        self.test_df.to_csv(self.test_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def create_validator(self, case_name):
        output_dir = self.root / case_name
        config = SimpleNamespace(
            valid_train_file_path=str(output_dir / 'validated' / 'train.csv'),
            valid_test_file_path=str(output_dir / 'validated' / 'test.csv'),
            invalid_train_file_path=str(output_dir / 'invalid' / 'train.csv'),
            invalid_test_file_path=str(output_dir / 'invalid' / 'test.csv'),
            drift_report_file_path=str(output_dir / 'drift_report' / 'report.yaml'),
        )
        artifact = DataIngestionArtifact(
            trained_filepath=str(self.train_path),
            test_filepath=str(self.test_path)
        )
        return DataValidation(artifact, config)

    def test_valid_data_is_written_to_validated_directory(self):
        result = self.create_validator('valid_case').initiate_data_validation()

        self.assertTrue(result.validation_status)
        self.assertTrue(Path(result.valid_train_file_path).exists())
        self.assertTrue(Path(result.valid_test_file_path).exists())
        self.assertIsNone(result.invalid_train_file_path)
        self.assertIsNone(result.invalid_test_file_path)
        self.assertTrue(Path(result.drift_report_file_path).exists())

    def test_schema_failure_is_written_to_invalid_directory(self):
        self.test_df.drop(columns=['Result']).to_csv(self.test_path, index=False)

        result = self.create_validator('invalid_case').initiate_data_validation()

        self.assertFalse(result.validation_status)
        self.assertIsNone(result.valid_train_file_path)
        self.assertIsNone(result.valid_test_file_path)
        self.assertTrue(Path(result.invalid_train_file_path).exists())
        self.assertTrue(Path(result.invalid_test_file_path).exists())
        self.assertIsNone(result.drift_report_file_path)

    def test_quality_checks_reject_nulls_and_values_outside_the_schema(self):
        validator = self.create_validator('quality_case')
        invalid_df = self.train_df.copy()
        invalid_df.loc[invalid_df.index[0], 'URL_Length'] = 99
        invalid_df.loc[invalid_df.index[1], 'Prefix_Suffix'] = None

        self.assertFalse(validator.validate_allowed_values(invalid_df))
        self.assertFalse(validator.validate_missing_values(invalid_df))


if __name__ == '__main__':
    unittest.main()
