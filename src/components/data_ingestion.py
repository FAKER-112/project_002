import os
import sys
import logging
import numpy as np
import pandas as pd
from zenml import step

class DataIngestion:
    def __init__(self) -> None:
        pass

    def get_data(self) -> pd.DataFrame:
         # Get src/ directory (one level up from components)
        base_dir = os.path.dirname(os.path.dirname(__file__))
        file_path = os.path.join(base_dir, "data", "olist_customers_dataset.csv")
        data = pd.read_csv(file_path)
        return data
def initate_data_ingestion():
    try:
        DataIngestor= DataIngestion()
        data=DataIngestor.get_data()
        return data

    except Exception as e:
        logging.error(e)
        raise e