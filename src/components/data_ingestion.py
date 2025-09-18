import os
import sys
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformationConfig,DataTransformation

@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join('artifacts','data.csv')
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion component")

        try:
            server = "SANKALP\SQLEXPRESS"
            database = "mlproject"
            driver = "ODBC Driver 17 for SQL Server"

            connection_url = f"mssql+pyodbc://@{server}/{database}?driver={driver}& trusted_connection=yes"
            engine = create_engine(connection_url)

            query = "SELECT * FROM stud"
            df=pd.read_sql(query,engine)

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Raw data ingested from SSMS")
            logging.info("Train test split started")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=40)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data splitted and saved in artifacts")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info(CustomException(e,sys))


if __name__ == "__main__":
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
            
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)
