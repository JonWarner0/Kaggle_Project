import sys
import numpy as np
import random as rand


def Shuffle(data):
    for i in range(len(data)):
        j = rand.randint(0, len(data)-1)
        temp = data[i]
        data[i] = data[j]
        data[j] = temp
    return data


def Standard(D):
    w = np.zeros(len(D[0][0]))
    T = 100
    r = 0.00001
    for t in range(T): 
        updated = False
        D = Shuffle(D)
        for x, y in D:
            if y * (w.T @ x) <= 0:
                w = w + r*y*x
                updated = True
        if not updated:
            print('converged')
            break
    return w


def Voted(D):
    w = np.zeros(len(D[0][0]))
    T = 100
    r = 0.001
    all_weights = []
    for t in range(T): 
        for x, y in D:
            if y * (w.T @ x) <= 0:
                w = w + r*y*x
                all_weights.append((w,1))
            else:
                wm, cm = all_weights[-1]
                all_weights[-1] = (wm, cm+1)
    return all_weights


def Average(D):
    w = np.zeros(len(D[0][0]))
    T = 100
    r = 0.00001
    a = 0
    for t in range(T): 
        update = False
        D = Shuffle(D)
        for x, y in D:
            if y * (w.T @ x) <= 0:
                w = w + r*y*x
                update = True
        a += w
        if not update:
            print('converged')
    return a


def RunTests(test, w, voted=False):
    correct = 0
    incorrect = 0
    for x, y in test:
        prediction = 0
        if voted:
            prediction = np.sign(sum([ci*(np.sign(wi.T @ x)) for wi, ci in w ]))
        else:
            prediction = np.sign(w.T @ x)

        if prediction == y:
            correct += 1
        else:
            incorrect += 1
    return correct, incorrect


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
    with open('PercepPredictions.csv', 'w+') as f:
        f.write('ID,Prediction\n')
        for p in predictions:
            f.write('{},{}\n'.format(p[0],p[1]))


if __name__ == '__main__':
    train = readTrainFile(sys.argv[1])
    test = readTestFile(sys.argv[2])
    
    function = sys.argv[3]
    w = None

    voted = False
    if function == '--standard':
        w = Standard(train)
    elif function == '--voted':
        w = Voted(train)
        voted = True
    elif function == '--avg':
        w = Average(train)
    else:
        print('Unknown command sequence. Use:\n --standard\n --voted\n --avg')
        exit()

    p = GeneratePredictions(test, w)
    OutputFile(p)

    # c_test, i_test = RunTests(test, w, voted)
    # print('test =>', function[2:], c_test, i_test, ' Error:', i_test/(c_test+i_test))
    # if not voted:
    #     print('Weights:', w)
    # else:
    #     for i in range(0,len(w), 20):
    #         print(w[i][0], w[i][1])