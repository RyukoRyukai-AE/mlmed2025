import pandas as pd
import yaml
import os

class Dataset:
    def __init__ (self, file_config:str):
        self.__path = file_config

    def get_path (self, file_name:str) -> str:
        data_path = ''

        with open(self.__path, 'r') as file:
            config = yaml.safe_load(file)

        data_path = config['paths']['data']
        data_path = os.path.join(data_path, file_name)

        return data_path

    def read_data (self, data_path:str, dfname:str) -> pd.DataFrame:
        df = pd.read_csv(data_path, header=None)
        df.dataframeName = dfname
        return df
    
    def display (self, df:pd.DataFrame):
        print(df.head(5))