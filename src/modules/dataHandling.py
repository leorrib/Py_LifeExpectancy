from src.tools.data_processing.dataHandler import DataHandler


class DataHandling():

    def __init__ (self, json, df):
        self.dh = json['data_handling']
        self.df = df

    def handle_data (self):

        cols_to_norm = self.df.columns.difference(self.dh['cols_not_to_normalize'])
        scaled_data = DataHandler.min_max_cols(self.df, cols_to_norm)

        DataHandler.get_strong_corr_predict_vars(scaled_data, self.dh['target_var'], 0.8)  

        scaled_data = scaled_data.drop(self.dh['cols_to_drop'], axis = 1)

        return scaled_data