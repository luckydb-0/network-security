from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    trained_filepath: str
    test_filepath: str