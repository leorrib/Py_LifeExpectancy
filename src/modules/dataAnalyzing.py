from src.tools.data_processing.dataLoader import DataLoader
from src.tools.data_processing.dataExplorer import DataExplorer
from src.tools.data_processing.dataHandler import DataHandler

class DataAnalyzing:

    def __init__ (self, json):
        self.dp = json['data_processing']

    def _loading_data(self):
        DataLoader(self.dp['toc_path']).diplay_toc()
        database = DataLoader(self.dp['db_path']).load_data()
        database = DataHandler(database).change_col_names(self.dp["cols_to_rename"])
        return database

    def _exploring_data(self, database):

        DataExplorer(database).find_values(None)
        database = DataHandler(database).drop_nas(
            self.dp['cols_to_drop_na'])

        DataExplorer(database).find_values(0)
        database = DataHandler(database).drop_values_from_cols(
            0, self.dp['cols_to_drop_zero'])

        database = DataHandler(database).factorize_vars(self.dp['vars_to_factorize'])

        de = DataExplorer(database)

        de.visualize_target_var(self.dp['data_vis_hist'])
                
        de.get_strong_corr(self.dp['target_var'], self.dp['strong_corr_cutoff'])

        de.plot_corr_heatmap()
            
        de.multiple_var_plot(self.dp['data_vis_strong_pos_corr'], self.dp['target_var'])

        de.multiple_var_plot(self.dp['data_vis_strong_neg_corr'], self.dp['target_var'])

        return database

    def analyze_data(self):
            df = self._loading_data()
            dataf = self._exploring_data(df)
            return dataf