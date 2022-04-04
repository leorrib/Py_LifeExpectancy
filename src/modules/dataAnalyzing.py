from src.tools.data_processing.dataLoader import DataLoader
from src.tools.data_processing.dataExplorer import DataExplorer

class DataAnalyzing:

    def __init__ (self, json):
        self.dp = json['data_processing']

    def _loading_data(self):
        DataLoader.diplay_toc(self.dp['toc_path'])
        database = DataLoader.load_data(self.dp['db_path'])
        database = DataLoader.change_col_names(self.dp["cols_to_rename"], database)
        return database

    def _exploring_data(self, database):

        DataExplorer.find_values(database, None)
        database = DataExplorer.drop_nas(database, cols_to_drop = self.dp['cols_to_drop_na'])

        DataExplorer.find_values(database, 0)
        database = DataExplorer.drop_values_from_cols(database, 0, self.dp['cols_to_drop_zero'])

        database = DataExplorer.factorize_var(database, self.dp['factorize_var'])

        hist_vis = self.dp['data_vis_hist']
        DataExplorer.visualize_target_var(
            database, self.dp['target_var'],  
            bins = hist_vis['bins'] , hue = hist_vis['hue'], 
            x_label = hist_vis['x_label'], y_label = hist_vis['y_label'], 
            labels = hist_vis['labels']
        )
                
        DataExplorer.get_strong_corr(database, self.dp['target_var'], 0.7)

        DataExplorer.plot_corr_heatmap(database)
            
        DataExplorer.multiple_var_plot(database, self.dp['data_vis_strong_pos_corr'], self.dp['target_var'])

        DataExplorer.multiple_var_plot(database, self.dp['data_vis_strong_neg_corr'], self.dp['target_var'])

        return database

    def analyze_data(self):
            df = self._loading_data()
            dataf = self._exploring_data(df)
            return dataf