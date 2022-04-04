from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureSelector():


    def correlation_analysis (df, target_var, X, Y):

        best_var = SelectKBest(score_func = f_regression, k = 'all')
        fit = best_var.fit(X, Y)
        fit.transform(X)

        data = {'Index': X.columns, 'Relevance': fit.scores_} 
        df = pd.DataFrame(data)
        df['Relevance'] = np.log(data['Relevance'])
        df.set_index('Index', inplace = True)

        fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (14, 7))
        bplot = sns.barplot(x = [i for i in range(len(fit.scores_))], y = fit.scores_, ax = ax1, color = 'grey')
        bplot.set_title('Correlation relevance distribution', fontdict={'fontsize':18}, pad=16)
        heatmap = sns.heatmap(df.sort_values(by='Relevance', ascending=False), vmin=3, vmax=9, annot=True, cmap='BrBG', ax = ax2)
        heatmap.set_title(f'Features Correlating with {target_var} (log scale)', fontdict={'fontsize':18}, pad=16)

        fig.tight_layout()
        plt.show()