import pandas as pd
import numpy as np
import seaborn as sns
import math
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class DataExplorer():


    def find_values(df, value):
        ''''
            Finds the desired value in columns
        '''

        for i in range(len(df.columns)):
            colname = df.columns[i]
            if value == None:
                S = df.iloc[:, i].isna().sum()
            else:
                S = (df.iloc[:, i] == value).sum()
            msg = f'The column {colname} has {S} values equal to {value}.'
            if S != 0:
                print(msg) 

    def drop_nas(df, cols_to_drop = []):
        ''''
            Drops entire columns, aside from rows with NAs
        '''

        df = df.drop(cols_to_drop, axis = 1)
        df = df.dropna(axis = 0)
        print(df.shape)
        return df

    def drop_values_from_cols(df, value, cols):
        ''''
            Drops rows with certain values in the provided columns
        '''

        for i in range(len(cols)):
            df = df.drop(df[df[cols[i]] == value].index)
        print(df.shape)
        return df

    def factorize_var(df, variable):
        df[variable] = pd.factorize(df[variable])[0]
        return df
    
    def divide_var_per_range(df):
        numero_de_faixas = 3
        n = int((max(df['Life_expectancy']) - min(df['Life_expectancy'])) / numero_de_faixas)
        my_range = range(int(min(df.Life_expectancy)), int(max(df.Life_expectancy)) + 2, n + 1)

        df['Life_expectancy_range'] = pd.cut(x = df.Life_expectancy, 
                                    bins = my_range, 
                                    labels = ['36-53', '54-71', '72-89'])

    def visualize_target_var(df, target_var, bins, hue, x_label, y_label, labels):
        histplt = sns.displot(df, 
                      x = target_var, 
                      bins = bins, 
                      legend = False, 
                      hue = hue, 
                      palette = ['skyblue', 'purple'])
        edges = [rect.get_x() for rect in histplt.ax.patches] + [histplt.ax.patches[-1].get_x() + histplt.ax.patches[-1].get_width()]
        # mids = [rect.get_x() + rect.get_width() / 2 for rect in histplt.ax.patches]
        histplt.ax.set_xticks(edges)
        histplt.set(xlabel = x_label, ylabel = y_label)
        plt.legend(loc = 'best', labels = labels)
        plt.show()

    def get_strong_corr(data, var_target, cutoff):
        df = data.corr(method = 'spearman')
        for j in range(len(df.columns)):
            for i in range(j, len(df)):
                if (abs(df.iloc[i, j]) > cutoff) and (i != j) and (df.columns[j] == var_target or df.index[i] == var_target):
                    print(f'Corr coef between {df.columns[j]} and {df.index[i]}: {df.iloc[i, j]}')

    def plot_corr_heatmap(df):
        cors = df.corr(method = 'spearman')
        plt.figure(figsize=(16, 8))
        upper = np.triu(np.ones_like(cors))
        heatmap = sns.heatmap(cors, vmin = -1, vmax = 1, annot = True, mask = upper, cmap = 'coolwarm')
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12)
        plt.show()

    def multiple_var_plot(df, lista_vars, target_var):

        row_num = math.ceil(len(lista_vars) / 2)
        
        cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
        fig, ax = plt.subplots(row_num, 2, figsize = (16, 8))
            
        num = 0
        if row_num == 1:
            while num < len(lista_vars):
                for j in range(0, 2):
                    sns.scatterplot(
                        data = df, x = lista_vars[num], y = target_var, 
                        hue = target_var, palette = cmap, ax = ax[j]
                    )
                    num = num + 1
                    if num == len(lista_vars): break
        else:
            while num < len(lista_vars):
                for i in range(0, row_num):
                    for j in range(0, 2):
                        sns.scatterplot(data = df, x = lista_vars[num], y = target_var, 
                                        hue = target_var, palette = cmap, ax = ax[i,j])
                        num = num + 1
                        if num == len(lista_vars): break
                    if num == len(lista_vars): break

        plt.show()


        