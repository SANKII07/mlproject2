import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features:pd.DataFrame):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            feature_names_path = os.path.join('artifacts','feature_names.pkl')

            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            feature_names = load_object(file_path = feature_names_path)

            features = features.reindex(columns=feature_names,fill_value=0)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
            
        except Exception as e:
            raise CustomException(e,sys)
        
    
class CustomData:
    def __init__(self,data:dict):
        self.data = data

    def get_data_as_dataframe(self):
        try:
            return pd.DataFrame([self.data])
        except Exception as e:
            raise CustomException(e,sys)        