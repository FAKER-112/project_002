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
        data =pd.read_csv('my-own-version\src\data\olist_customers_dataset.csv')
        return data

def initate_data_ingestion(self):
    try:
        DataIngestion=DataIngestion()
        data=DataIngestion.get_data()
        return data

    except Exception as e:
        logging.error(e)
        raise e