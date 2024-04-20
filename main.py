import pandas as np
from perceptron import Perceptron as pr
import statsmodels.tools as sm


def quesD():
    dataset = [[-2, -1, -1], [0, 0, 1], [2, 1, 1], [1, 2, 1], [-2, 2, -1], [-3, 0, -1]]
    df = np.DataFrame(dataset, columns=["x1", "x2", "y"])
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    x = sm.add_constant(x)
    perc = pr()
    perc.fit(x, y)
    print("the weights of the data set from question A is:\n",perc.weights_)
    print("the score is:", perc.score(x,y))

#main
quesD()