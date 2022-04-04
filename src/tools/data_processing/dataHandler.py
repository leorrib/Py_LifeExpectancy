from sklearn.preprocessing import MinMaxScaler
from sklearn_pandas import DataFrameMapper
import pandas as pd

class DataHandler():

    def min_max_cols(dataf, cols):
        df = dataf[cols]
        mapper = DataFrameMapper([(df.columns, MinMaxScaler(feature_range = (0, 1)))])
        s_data = mapper.fit_transform(df.copy(), 4)
        scaled_data = pd.DataFrame(s_data, index = df.index, columns = df.columns)
        non_norm_cols = list(set(dataf.columns) - set(df.columns))
        for i in range(len(non_norm_cols)):
            scaled_data[non_norm_cols[i]] = dataf[non_norm_cols[i]]

        return scaled_data

    def get_strong_corr_predict_vars(df, target_var, cutoff):
        corr_mat = df.corr(method = 'spearman')
        for j in range(len(corr_mat.columns)):
            for i in range(j, len(corr_mat)): 
                if (abs(corr_mat.iloc[i, j] > cutoff) and (i != j) and 
                    corr_mat.columns[j] != target_var and corr_mat.index[i] != target_var):
                    print(f"Corr coef between {corr_mat.columns[j]} and {corr_mat.index[i]}: {corr_mat.iloc[i, j]}")
                    
