import sys
import numpy as np
import random as rand
import math


def Shuffle(data):
    for i in range(len(data)):
        j = rand.randint(0, len(data)-1)
        temp = data[i]
        data[i] = data[j]
        data[j] = temp
    return data


def Gamma(g, t, d):
    return g/(1+(g/d)*t)


def MAP(s, w, v, M):
    sigmoid = 1/(1+math.exp(-1*s[1]*w@s[0]))
    return -1*(1-sigmoid)*M*s[1]*s[0] + w/(v**2)


def MLE(s, w, v, M):
    z = -1*s[1]*w@s[0]
    q = 0
    try:
        q = math.exp(z)
    except Exception as e:
        print(e)
        print('z: ', z)
        print('w: ', w)
        print('x: ', s[0])
        exit()
    sigmoid = 1/(1+q)
    return -1*(1-sigmoid)*M*s[1]*s[0]



def SGD(Data, g, d, var, func=MAP):
    w = np.zeros(len(Data[0][0]))
    M = len(Data)
    for t in range(100):
        data = Shuffle(Data)
        for ex in data:
            g = Gamma(g,t,d)
            w = w - g*func(ex,w,var,M)
    return w


def readTrainFile(input_file):
    values = []
    with open(input_file) as f:
        for line in f:
            v = line.strip().split(',')
            data = np.array([1]+[float(n) for n in v[:-1]])
            label = 2*int(v[-1])-1
            values.append((data,label))
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
    with open('resultsLogRegression.csv', 'w+') as f:
        f.write('ID,Prediction\n')
        for p in predictions:
            f.write('{},{}\n'.format(p[0],p[1]))


if __name__ == '__main__':
    train = readTrainFile(sys.argv[1])
    test = readTestFile(sys.argv[2])
    estimation = sys.argv[3]

    func = None
    if estimation == '--map':
        func = MAP
    elif estimation == '--mle':
        func = MLE
    else:
        print('Unknown command sequence')
        exit()

    var = [0.01,0.1,0.5,1,3,5,10,100]
    gamma = 0.001 # MLE 0.0001
    d = 0.000025 # MLE: 0.5

    model = SGD(train, gamma, d, var[1], func=func)
    p = GeneratePredictions(test, model)
    OutputFile(p)
