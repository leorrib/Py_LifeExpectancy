from yellowbrick.regressor import ResidualsPlot
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

class Visualizer():

    def visualize_residue_spread(model, X_train, X_test, Y_train, Y_test):
        visualizer = ResidualsPlot(model)
        visualizer.fit(X_train, Y_train)
        visualizer.score(X_test, Y_test)
        visualizer.show()

    def _abline(slope, intercept):
        plt.style.use('ggplot')
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, '--', color = 'red')

    def visualize_residue_line(Y_test, Y_pred):
        plt.scatter(Y_test, Y_pred, color = 'black')
        Visualizer._abline(1, 0)
        plt.xlabel('Value computed for target variable')
        plt.ylabel('Real value of the target variable')
        plt.show()