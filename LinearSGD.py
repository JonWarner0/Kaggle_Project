import numpy as np
from numpy import linalg as LA
import sys
import math

class entry:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y


def Calc_Gradient(X,w,j):
    total = 0
    for ex in X:
        total += (ex.y - w.T @ ex.x) * ex.x[j]
    return -total


def GradientDescent(X,w,r,threshold):
    while True:
        for ex in X:
            new_w = []
            for j in range(len(w)):
                v = w[j] + r*(ex.y - w.T @ ex.x)*ex.x[j]
                print(v)
                if math.isnan(v):
                    exit()
                new_w.append(v)
            error = LA.norm(w-new_w, ord=2)
            #print(error)
            if error <= threshold:
                return w
            w = np.array(new_w)


def Eval(f, tests):
    sq_error = 0
    for t in tests:
        prediction = f @ t.x
        sq_error += (prediction-t.y)**2
    return sq_error/2


def ReadFile(file):
    vals = []
    with open(file,'r') as f:
        for line in f:
            ex = line.strip().split(',')
            x = np.array([float(v) for v in ex[:-1]])
            np.insert(x,0,1)
            y = float(ex[-1])
            vals.append(entry(x,y))
    return vals


def Calculate_Optimal(train):
    x = []
    y = []
    for t in train:
        x.append(t.x)
        y.append(t.y)
    X = np.array(x)
    Y = np.array(y)
    return LA.inv(X.T @ X) @ (X.T @ Y)


def readTrainFile(input_file):
    values = []
    with open(input_file) as f:
        for line in f:
            v = line.strip().split(',')
            data = np.array([1]+[float(n) for n in v[:-1]])
            label = 2*int(v[-1])-1
            values.append(entry(data,label))
    return values


def readTestFile(file):
    values = []
    with open(file) as f:
        for line in f:
            v = line.strip().split(',')
            data = np.array([1]+[float(n) for n in v[1:]])
            values.append(data)
    return values


def GeneratePredictions(test, w):
    p = []
    for i in range(len(test)):
        p.append((i+1, w @ test[i]))
    return p


def OutputFile(predictions):
    with open('resultsLinearSGD.csv', 'w+') as f:
        f.write('ID,Prediction\n')
        for p in predictions:
            f.write('{},{}\n'.format(p[0],p[1]))


if __name__ == '__main__':  
    training_f = sys.argv[1]
    testing_f = sys.argv[2]

    train = readTrainFile(training_f)
    tests = readTestFile(testing_f)

    w = np.zeros(len(train[0].x))
    r = 0.0000125
    et = 0.00001

    #w_star = GradientDescent(train, w, r, et)
    w_star = Calculate_Optimal(train)
    p = GeneratePredictions(tests, w_star)
    OutputFile(p)

