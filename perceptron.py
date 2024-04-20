import numpy as np
import statsmodels.tools as sm
import pandas as pd


class Perceptron:
    def __init__(self):
        self.weights_ = None

    def fit(self, x, y):
        x= sm.add_constant(x)
        self.weights_ = np.zeros(x.shape[1])
        for i in range(0,100):
            i=0
            for gt in y:
                calc = self.weights_ @ np.transpose(x.iloc[i,:])
                pred = -1 if calc <0 else 1
                self.weights_ = np.subtract(self.weights_, (gt-pred)/2*x-x.iloc[i,:])
        return True




