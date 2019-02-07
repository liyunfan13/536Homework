import numpy as np
import random
import math

def generateData(k, m):
    random.seed()
    data = []
    for j in range(m):
        sample = np.zeros(k+1, dtype=int)
        seed1 = random.random()
        sample[0] = 1 if seed1>0.5 else 0
        for i in range(1, k):
            seed2 = random.random()
            sample[i] = sample[i-1] if seed2 > 0.25 else (1-sample[i-1])
        weights = np.zeros(k-1)
        weightsum = 0.0
        for i in range(k-1):
            weightsum += math.pow(0.9, i+2)

        for i in range(k-1):
            weights[i] = math.pow(0.9, i+2)/weightsum
        sum = 0.0
        for i in range(k-1):
            sum += weights[i]*sample[i+1]
        if sum>=0.5:
            sample[k] = sample[0]
        else:
            sample[k] = 1-sample[0]
        data.append(sample)

    return data

def partition(data, x):
    num_of_train = int(x*len(data))
    num_of_val = len(data)-num_of_train
    train_set = []
    val_set = []
    assert type(data)==list, "The dataset should be a list."
    for i in range(num_of_val):
        seed = random.randint(0,len(data)-1)
        d = data.pop(seed)
        val_set.append(d)
    train_set = data

    return train_set, val_set


data = generateData(5,10)
train, val = partition(data,0.8)
for i in train:
    print(i)
print('=======')
for i in val:
    print(i)

