import pandas as np
from perceptron import Perceptron as pr
#main
dataset = [[-2,-1,-1],[0,0,1],[2,1,1],[1,2,1],[-2,2,-1],[-3,0,-1]]
df = np.DataFrame(dataset,columns=["x1","x2","y"])
x= df.iloc[:,0:-1]
y = df.iloc[:,-1]
perc = pr()
perc.fit(x,y)
print(perc.weights_)

