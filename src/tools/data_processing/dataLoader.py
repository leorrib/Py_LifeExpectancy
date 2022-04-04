from IPython.display import Image, display
import pandas as pd

class DataLoader():

    def diplay_toc(path):
        img = Image(path)
        display(img)

    def load_data(path, blank_replacement = '_'):
        database = pd.read_csv(path)
        database = database.rename(columns = lambda x: x.strip())
        database = database.rename(columns = lambda x: x.capitalize().replace(' ', blank_replacement))
        print(f'Dimensionality: {database.shape}')
        print(f'Variables:\n {database.columns}')
        return database

    def change_col_names(cols_to_rename, df):

        df.rename(columns = cols_to_rename, inplace = True)
        print(f'Variables:\n {df.columns}')
        return df
