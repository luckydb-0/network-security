from dataclasses import dataclass
from typing import Optional

@dataclass
class DataIngestionArtifact:
    trained_filepath: str
    test_filepath: str

@dataclass
class DataValidationArtifact:
    validation_status: bool
    valid_train_file_path: Optional[str]
    valid_test_file_path: Optional[str]
    invalid_train_file_path: Optional[str]
    invalid_test_file_path: Optional[str]
    drift_report_file_path: Optional[str]
