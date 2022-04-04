import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

class Algorithms():

    def train_Linear_Regression_model(X_train, Y_train):
        model = LinearRegression()
        model.fit(X_train, Y_train)
        r2 = model.score(X_train, Y_train)
        print(f'R-squared value (training): {r2}')
        return model

    def test_Linear_Regression_model(X_test, Y_test, model):
        Y_pred = model.predict(X_test)

        r2 = model.score(X_test, Y_test)
        mrse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        print(f'R-squared value: {r2}')
        print(f'Root mean squared error: {mrse}')
        print(f'Normalized root mean squared error: {mrse / (max(Y_pred) - min(Y_pred))}')
        return Y_pred

    def train_Random_Forest_model(X_train, Y_train):
        model = RandomForestRegressor()
        model.fit(X_train, Y_train)
        r2 = model.score(X_train, Y_train)
        print(f'R-squared value (training): {r2}')
        return model

    def test_Random_Forest_model(X_test, Y_test, model):
        Y_pred = model.predict(X_test)

        r2 = model.score(X_test, Y_test)
        mrse = np.sqrt(mean_squared_error(Y_test, Y_pred))
        print(f'R-squared value: {r2}')
        print(f'Root mean squared error: {mrse}')
        print(f'Normalized root mean squared error: {mrse / (max(Y_pred) - min(Y_pred))}')
        return Y_pred

    def cross_val_score_model(X, Y, model):
        if model == 'LinearRegression':
            model = LinearRegression()
        elif model == 'RandomForest':
            model = RandomForestRegressor()

        n_folds = 5
        kfold = KFold(n_splits = n_folds, shuffle = True)

        # score = cross_val_score(modelo, X, Y, cv = kfold)
        r2 = cross_val_score(model, X, Y, cv = kfold, scoring = 'r2')
        rmse = cross_val_score(model, X, Y, cv = kfold, scoring = 'neg_root_mean_squared_error')

        print(f'R-squared value: {np.mean(r2)}')
        print(f'Root mean squared error: {-np.mean(rmse)}')
        print(f'Normalized root mean squared error: {-np.mean(rmse / (max(Y) - min(Y)))}')
