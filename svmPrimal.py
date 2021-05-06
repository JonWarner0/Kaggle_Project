import sys
import numpy as np
import random as rand

#--- Constants ---
G_INIT = 0.1
T = 100
C_set = [(100/873), (500/873), (700/873)]


def Shuffle(data):
	for i in range(len(data)):
		j = rand.randint(0,len(data)-1)
		temp = data[i]
		data[i] = data[j]
		data[j] = temp
	return data


def Gamma(t):
	return G_INIT/(1+(G_INIT/C)*t)


def StochasticSVM(C,N,S):
	w_not = np.zeros(len(S[0][0])-1)
	w = np.zeros(len(S[0][0]))
	for t in range(T):
		data = Shuffle(S)
		gamma = Gamma(t)
		for ex in data:
			if ex[1] * w.T @ ex[0] <= 1:
				w_temp = np.append(w_not,0) 
				w = w-gamma*w_temp+gamma*C*N*ex[1]*ex[0]
			else:
				w_not = (1-gamma)*w_not
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
    with open('resultSVM.csv', 'w+') as f:
        f.write('ID,Prediction\n')
        for p in predictions:
            f.write('{},{}\n'.format(p[0],p[1]))


if __name__ == '__main__':
	train = readTrainFile(sys.argv[1])
	tests = readTestFile(sys.argv[2])
	C = C_set[int(sys.argv[3])]
	w = StochasticSVM(C,len(train),train)
	p = GeneratePredictions(tests, w)
	OutputFile(p)
