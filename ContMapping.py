import sys
import random as rand

if __name__ == '__main__':
    features = {i:set() for i in range(14)}
    num_idx = {0,2,4,10,11,12}
    first_line = True
    DATA = []

    with open(sys.argv[1], 'r+') as f:
        for line in f:
            if first_line:
                first_line = False 
                continue
            term = line.strip().split(',')
            DATA.append(term)
            for i in range(len(term)-1):
                if i in num_idx:
                    continue
                features[i].add(term[i])

    cont_mapping = {i:None for i in range(14)}
    for i in features.keys():
        cont_mapping[i] = {v : str(rand.randint(-1000,1000)) for v in features[i]}
    
    with open('mappedTrainData.csv', 'w+') as f:
        for s in DATA:
            for i in range(len(s)-1):
                if i not in num_idx:
                    s[i] = cont_mapping[i][s[i]]
            p = ','.join(s)
            f.write('{}\n'.format(p))

    first_line = True
    TEST = []
    with open(sys.argv[2], 'r+') as f:
        for line in f:
            if first_line:
                first_line = False 
                continue
            l = line.strip().split(',')
            TEST.append(l)

    not_seen = dict()
    with open('mappedTestData.csv', 'w+') as f:
        for s in TEST:
            for i in range(1,len(s)):
                if i-1 not in num_idx:
                    if s[i] in cont_mapping[i-1].keys():
                        s[i] = cont_mapping[i-1][s[i]]
                    elif s[i] in not_seen.keys():
                        s[i] = not_seen[s[i]]
                    else:
                        n = str(rand.randint(-1000,1000))
                        not_seen[s[i]] = n
                        s[i] = n
            p = ','.join(s)
            f.write('{}\n'.format(p))
            