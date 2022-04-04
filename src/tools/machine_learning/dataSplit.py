from sklearn.model_selection import train_test_split

class DataSplit():

    def train_test_split(variables, target, test_size):
        return train_test_split(variables, target, test_size = test_size)