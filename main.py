import pandas as pd
from perceptron import Perceptron as pr
import statsmodels.tools as sm


def questionD():
    print("question E3:\n")
    dataset = [[-2, -1, -1], [0, 0, 1], [2, 1, 1], [1, 2, 1], [-2, 2, -1], [-3, 0, -1]]
    df = pd.DataFrame(dataset, columns=["x1", "x2", "y"])
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]
    x = sm.add_constant(x)
    perc = pr()
    perc.fit(x, y)
    print("the weights of the data set from question A is:\n",perc.weights_)
    print("the score is:", perc.score(x,y)*100,"%")

def questionE():
    print("question E3:\n")
    df = pd.read_csv('Processed Wisconsin Diagnostic Breast Cancer.csv', delimiter= ',')
    #prepare the data base to precptron
    train_set, test_set = splitDataSet(df)
    x_train_set,y_train_set = splitIntoXandY(train_set)
    x_test_set, y_test_set = splitIntoXandY(test_set)
    y_train_set = zeroToMinosOne(y_train_set)
    y_test_set = zeroToMinosOne(y_test_set)

    df_perc = pr()
    df_perc.fit(x_train_set,y_train_set)
    score_test = df_perc.score(x_test_set,y_test_set)
    score_train = df_perc.score(x_train_set,y_train_set)
    print("Perceptron score is: ",score_test*100,"%")
    print("error rate on training set is:", 1-score_train)
    print("total error rate is:", 1 - 0.8*score_train- 0.2 *score_test)

#split df to paramelts and labels
def splitIntoXandY(df):
    x,y= df.iloc[:,0:-1],df.iloc[:,-1]
    x = sm.add_constant(x)
    return x,y

# change the zero lable to -1
def zeroToMinosOne(df):
    df = df.apply(lambda x : -1 if x ==0 else x)
    return df
# shaffel and split the data to test and train
def splitDataSet(df):
    df_shaffeld = df.sample(frac= 1 , random_state = 42)
    split_index = int(0.8* len(df_shaffeld))
    return df_shaffeld.iloc[0:split_index],df_shaffeld[split_index:-1]


#main
questionD()
questionE()
