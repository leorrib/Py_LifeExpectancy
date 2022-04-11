from src.tools.data_processing.dataHandler import DataHandler
from src.tools.data_processing.dataExplorer import DataExplorer


class DataHandling():

    def __init__ (self, json, df):
        self.dh = json['data_handling']
        self.df = df

    def handle_data (self):

        scaled_data = DataHandler(self.df).min_max_cols(self.dh['cols_to_normalize'])

        DataExplorer(scaled_data).get_strong_corr_predict_vars(self.dh['target_var'], 0.8)  

        scaled_data = scaled_data.drop(self.dh['cols_to_drop'], axis = 1)

        return scaled_data