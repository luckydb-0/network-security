import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from src.entity.artifact_entity import DataValidationArtifact

data_ingestion_module = ModuleType('src.components.data_ingestion')
data_ingestion_module.DataIngestion = object
with patch.dict(sys.modules, {'src.components.data_ingestion': data_ingestion_module}):
    import main


class TestTrainingPipeline(unittest.TestCase):
    @patch('builtins.print')
    @patch.object(main, 'DataTransformation')
    @patch.object(main, 'DataTransformationConfig')
    @patch.object(main, 'DataValidation')
    @patch.object(main, 'DataValidationConfig')
    @patch.object(main, 'DataIngestion')
    @patch.object(main, 'DataIngestionConfig')
    @patch.object(main, 'TrainingPipelineConfig')
    def test_transformation_is_skipped_when_validation_fails(
        self,
        training_pipeline_config,
        data_ingestion_config,
        data_ingestion,
        data_validation_config,
        data_validation,
        data_transformation_config,
        data_transformation,
        _print,
    ):
        ingestion_artifact = object()
        validation_artifact = DataValidationArtifact(
            validation_status=False,
            valid_train_file_path=None,
            valid_test_file_path=None,
            invalid_train_file_path='invalid/train.csv',
            invalid_test_file_path='invalid/test.csv',
            drift_report_file_path=None,
        )
        data_ingestion.return_value.initiate_data_ingestion.return_value = ingestion_artifact
        data_validation.return_value.initiate_data_validation.return_value = validation_artifact

        result = main.run_pipeline()

        self.assertIs(result, validation_artifact)
        training_pipeline_config.assert_called_once_with()
        data_ingestion_config.assert_called_once_with(training_pipeline_config.return_value)
        data_validation_config.assert_called_once_with(training_pipeline_config.return_value)
        data_transformation_config.assert_not_called()
        data_transformation.assert_not_called()


if __name__ == '__main__':
    unittest.main()
