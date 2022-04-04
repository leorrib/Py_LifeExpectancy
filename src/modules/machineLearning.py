from src.tools.machine_learning.featureSelector import FeatureSelector
from src.tools.machine_learning.dataSplit import DataSplit
from src.tools.machine_learning.algorithms import Algorithms
from src.tools.machine_learning.visualizer import Visualizer
import warnings
warnings.filterwarnings('ignore')

class MachineLearning:

    def __init__ (self, json, df):
        self.ml = json['machine_learning']
        self.df = df
        (self.X, self.Y, self.X_train, self.X_test, self.Y_train, self.Y_test) = self._prepare_data()

    def _prepare_data(self):

        last_col = len(self.df.columns) - 1
        X = self.df.iloc[:, 0:last_col]
        Y = self.df.iloc[:, last_col]
        FeatureSelector.correlation_analysis(self.df, self.ml['target_var'], X, Y)

        X = X.drop(self.ml['cols_to_drop'], axis = 1)

        X_train, X_test, Y_train, Y_test = DataSplit.train_test_split(X, Y,self.ml['df_train_size'])

        return (X, Y, X_train, X_test, Y_train, Y_test)

    def _build_linear_model(self):

        print('\nLinear Model data \n')

        model = Algorithms.train_Linear_Regression_model(self.X_train, self.Y_train)

        Y_pred = Algorithms.test_Linear_Regression_model(self.X_test, self.Y_test, model)

        Visualizer.visualize_residue_spread(model, self.X_train, self.X_test, self.Y_train, self.Y_test)

        Visualizer.visualize_residue_line(self.Y_test, Y_pred)

        return model

    def _build_cross_val_linear_model(self):

        print('\nCross validation - Linear Model data \n')

        model = Algorithms.cross_val_score_model(self.X, self.Y, 'LinearRegression')

        return model

    def _build_random_forest_model(self):

        print('\nRandom Forest model data \n')

        model = Algorithms.train_Random_Forest_model(self.X_train, self.Y_train)

        Y_pred = Algorithms.test_Random_Forest_model(self.X_test, self.Y_test, model)

        Visualizer.visualize_residue_spread(model, self.X_train, self.X_test, self.Y_train, self.Y_test)

        Visualizer.visualize_residue_line(self.Y_test, Y_pred)

        return model

    def _build_cross_val_random_forest_model(self):

        print('\nCross validation - Random Forest model data \n')

        model = Algorithms.cross_val_score_model(self.X, self.Y, 'RandomForest')

        return model

    def build_ml_models(self):
        model1 = self._build_linear_model()
        model2 = self._build_cross_val_linear_model()
        model3 = self._build_random_forest_model()
        model4 = self._build_cross_val_random_forest_model()
        return (model1, model2, model3, model4)