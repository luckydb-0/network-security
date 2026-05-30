import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.entity.artifact_entity import DataValidationArtifact


class TestDataTransformation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.train_path = self.root / 'validated' / 'train.csv'
        self.test_path = self.root / 'validated' / 'test.csv'
        self.train_path.parent.mkdir()

        pd.DataFrame({
            'feature_a': [1, 2, None, 4],
            'feature_b': [1, 2, 3, 4],
            'Result': [-1, 1, -1, 1],
        }).to_csv(self.train_path, index=False)
        pd.DataFrame({
            'feature_a': [None, 3],
            'feature_b': [2, 3],
            'Result': [-1, 1],
        }).to_csv(self.test_path, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_knn_imputer_fills_missing_feature_values(self):
        validation_artifact = DataValidationArtifact(
            validation_status=True,
            valid_train_file_path=str(self.train_path),
            valid_test_file_path=str(self.test_path),
            invalid_train_file_path=None,
            invalid_test_file_path=None,
            drift_report_file_path=None,
        )
        transformation_config = SimpleNamespace(
            transformed_train_file_path=str(self.root / 'transformed' / 'train.npy'),
            transformed_test_file_path=str(self.root / 'transformed' / 'test.npy'),
            transformed_object_file_path=str(self.root / 'transformed_object' / 'preprocessing.pkl'),
        )

        result = DataTransformation(
            validation_artifact,
            transformation_config,
        ).initiate_data_transformation()

        transformed_train = np.load(result.transformed_train_file_path)
        transformed_test = np.load(result.transformed_test_file_path)

        self.assertEqual(transformed_train.shape, (4, 3))
        self.assertEqual(transformed_test.shape, (2, 3))
        self.assertFalse(np.isnan(transformed_train[:, :-1]).any())
        self.assertFalse(np.isnan(transformed_test[:, :-1]).any())
        self.assertEqual(set(transformed_train[:, -1]), {0, 1})
        self.assertTrue(Path(result.transformed_object_file_path).exists())


if __name__ == '__main__':
    unittest.main()
