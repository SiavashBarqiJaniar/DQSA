# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:43:42 2019

@author: Siavash Barqi
"""

#Imports
from numpy import *
import numpy as np
from Env import Env
import plot as plt
from keras.models import load_model
from math import *
from numpy.random import choice

#Parameters
N = 3
K = 2
T = 50

def experience(env , DQN1):
    data = []
    throughput = 0
    state = env.reset()
    a = np.zeros(N, dtype=int)
    
    for t in range(T):
        data.append(t)
        data[t] = []
        for n in range(N):
            x = (state[n])[newaxis]
            Q = DQN1.predict(x)
            Q = Q[0]
            print(state[n], ': ', Q)
            p = ActionDistribution(env.K , Q , path)
                    
            list_of_candidates = range(K + 1)
            nn = sum(p)
            probability_distribution = [x/nn for x in p]
            number_of_items_to_pick = 1
            
            aMax = choice(list_of_candidates, number_of_items_to_pick,
                            p=probability_distribution)

            a[n] = aMax
            #aMax = argmax(p)
            #prob = .9
            """ran = random.uniform(0.0 , 1.0)
            if ran < prob:
                a[n] = aMax
            else:
                ran = random.randint(K) # random int between 0 and K-1
                if ran < aMax:
                    a[n] = ran
                else:
                    a[n] = ran + 1"""
        
        done = False
        if t==T-1:
            done = True
        _, action, state, _ = env.step(a, done)
        
        for i in range(N):
            if state[i][-1] == 1:
                throughput += 1
            data[t].append(action[i])
        
    throughput /= (T*K)
    #print('ex: ', data)
    return throughput , data

def ActionDistribution(K , Q , path, *con):
    alpha = 0
    beta = 20
    s = 0
    p = []
    count = 0
    check = False
    for a in range(K+1):
        if Q[a] >= 35:
            f = open(path + "explosion.txt" , "a")
            f.write(str(it)+': \r\n')
            f.write(str(Q) + '\r\n')
            #Q = [x/(it/1000 + 1) for x in Q]
            f.write(str(Q) + '\r\n')
            check = True
            count += 1
            f.close()
        s += exp(beta*Q[a])

    for a in range(K + 1):
        p.append( (((1 - alpha) * exp(beta * Q[a])) / s) + alpha / (K + 1) )
            
    return p

#Body
### Loading DQN's
DQN1 = load_model('DQN1.h5')

env2 = Env(N, K, 'competitive')
path = 'results/competitive/11/'

while True:
    FigNum = 0
    #s = input("Exit?[y/n]: ")
    #if s=='y':
    #    break
    
    for it in range(30):
        thr , data = experience(env2, DQN1)
        FigNum += 1
        print('throughput at experience #' , it+1 , ' : ' , "% 12.2f" % (100*thr))
        
        #figure
        plt.show(data, thr, it, path)
        dd = data[-1]
    
        if dd != []:
            print(dd)
    break
        
