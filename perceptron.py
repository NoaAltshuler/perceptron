import numpy as np
import statsmodels.tools as sm


class Perceptron:
    def __init__(self):
        self.weights_ = None

    def fit(self, x, y):
        x= sm.add_constant(x)
        self.weights_ = np.zeros(x.shape[1])
        d = [(row,val)for row,val in zip(x,y)]
        for i in range(0,100):
            for pair in d:
                calc = self.weights_ @ np.transpose(pair[0])
                pred = -1 if calc <0 else 1
                self.weights_ = np.subtract(self.weights_, (pair[1]-pred)/2*pair[0])



