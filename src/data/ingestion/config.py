from dataclasses import dataclass

@dataclass
class IngestionConfig:
    input_dir: str = "data/interim/extracted_mat"
    output_dir: str = "data/ingested"
    
    index_file: str = "index.csv"
    patient_map_file: str = "patient_map.json"
    
    allowed_labels: tuple = (1, 2, 3)
    label_map: dict = None