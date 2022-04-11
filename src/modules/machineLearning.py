from src.tools.machine_learning.featureSelector import FeatureSelector
from src.tools.machine_learning.dataSplit import DataSplit
from src.tools.machine_learning.algorithms import Algorithms
from src.tools.machine_learning.visualizer import Visualizer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class MachineLearning:

    def __init__ (self, json, df):
        self.ml = json['machine_learning']
        self.df = df
        (self.X, self.Y, self.X_train, self.X_test, self.Y_train, self.Y_test) = self._prepare_data()

    def _prepare_data(self):

        X = self.df[self.df.columns.difference([self.ml['target_var']])]
        Y = self.df[self.ml['target_var']]
        ds = DataSplit(self.ml['df_train_size'])
        (X_train, X_test, Y_train, Y_test) = ds.train_test_split(X, Y)

        fs = FeatureSelector(self.ml['target_var'], X_train, Y_train)
        fs.correlation_analysis_KBest(log_scale = True)

        X = X.drop(self.ml['cols_to_drop'], axis = 1)

        return (X, Y, X_train, X_test, Y_train, Y_test)

    def _build_linear_regression_model(self):
        
        print('\nLinear Model data \n')

        data = Algorithms(self.X_train, self.X_test, self.Y_train, self.Y_test)

        lr_tool = LinearRegression()
        print('\n1 - Cross val\n')
        data.cross_val_score_regression(lr_tool)

        print('\n2 - The model\n')
        results_lr_model = data.linear_regression_model()
        lr_model = results_lr_model['model']
        Visualizer(lr_tool).visualize_residue_spread(self.X_train, self.X_test, self.Y_train, self.Y_test)

        Y_pred_lr = results_lr_model['Y_pred']
        Visualizer(lr_tool).visualize_residue_line(self.Y_test, Y_pred_lr)

        return lr_model

    def _build_random_forest_model(self, ntrees):

        print('\nRandom Forest model data \n')

        data = Algorithms(self.X_train, self.X_test, self.Y_train, self.Y_test)
        print('\n1 - Cross val\n')
        rfr_tool = RandomForestRegressor(n_estimators = ntrees)
        data.cross_val_score_regression(rfr_tool)
        print('\n2 - The model\n')
        results_rfr_model = data.random_forest_regressor_model()

        rfr_model = results_rfr_model['model']
        Visualizer(rfr_tool).visualize_residue_spread(
            self.X_train, self.X_test, self.Y_train, self.Y_test)

        Y_pred_lr = results_rfr_model['Y_pred']
        Visualizer(rfr_tool).visualize_residue_line(self.Y_test, Y_pred_lr)
        return rfr_model

    def build_ml_models(self):
        model1 = self._build_linear_regression_model()
        model2 = self._build_random_forest_model(ntrees = self.ml["rf_ntrees"])
        return model1, model2