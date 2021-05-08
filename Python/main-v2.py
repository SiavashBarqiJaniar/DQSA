# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 11:43:42 2019

@author: Siavash Barqi
"""

#Imports
from numpy import *
import numpy as np
from Env import Env
from keras.models import Sequential
from keras.layers import Dense , Activation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from keras.layers import Input, Embedding, LSTM, Dense, add, average, concatenate, subtract, Reshape
from keras.models import Model
from math import *

#Variables
N = 3
K = 2
T = 50
M = 1
epsilon = 0.2
prb = 1 - epsilon
iteration = 10000

#Funcs
def iterate(M , T , scheme , env , prb):
    #DQN
    #1
    inp = Input(shape=(2*K+2,), dtype=float32)
    r = Reshape((1, -1))(inp)
    lstm = LSTM(100 , return_sequences=False, activation='relu')(r)
    v = Dense(20 , activation='relu')(lstm)
    v = Dense(20 , activation='relu')(v)
    OutputLayer = Dense(K+1)(v)
    
    #2
    inp2 = Input(shape=(2*K+2,), dtype=float32)
    r2 = Reshape((1, -1))(inp2)
    lstm2 = LSTM(100 , return_sequences=False, activation='relu')(r2)
    v2 = Dense(20 , activation='relu')(lstm2)
    v2 = Dense(20 , activation='relu')(v2)
    OutputLayer2 = Dense(K+1)(v2)
    
    DQN1 = Model(inputs = inp , outputs = OutputLayer)
    DQN2 = Model(inputs = inp2 , outputs = OutputLayer2)
    
    DQN1.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])
    
    DQN2.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mse'])
    
    print('Initialization')
    esw = array([1 , 0 , 0 , 0 , 0 , 0])[newaxis]
    dhg = array([0])[newaxis]
    dhgf = array([0,0,0])[newaxis]
    DQN1.fit(esw , dhgf , epochs=8, batch_size=4 , verbose = 0)
    print(DQN1.predict(esw))
    
    print('START')
    
    N = env.N
    output = ''
    excc = ""
    MA = 0
    for it in range(iteration):
        maxx = 0
        tq = ndarray(shape = (N , K + 1))
        V = ndarray(shape = (N , K + 1))
        OptA = ndarray(shape = (N , 2*K + 2))
        Q = ndarray(shape = (M*T*N , K + 1))
        x = ndarray(shape = (M*T*N , 2*K + 2) , dtype = int)
        time = -1

        for yy in range(15):
            output += '|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\r\n'
        output += 'at iteration ' + str(it) + '\r\n'
        
        for episode in range(M):
            throughput = 0
            data = []
            env.reset()
            
            for t in range(T):
                tprime = t
                output += '  TIME : ' + str(t) + ':\r\n'
                data.append([])
                data[t] = []

                for n in range(N):
                    time += 1
                    if it < 200:
                        xx = copy(env.observe(n))
                        x[time] = copy(xx)
                        tq[n] = DQN1.predict((xx)[newaxis])
                        rand = np.random.randint(N)
                        #print(rand)
                        if rand%N <= env.K:
                            a = rand
                        else:
                            a = 0
                        
                    else:
                        xx = copy(env.observe(n))
                        x[time] = copy(xx)
                        tq[n] = DQN1.predict((xx)[newaxis])
                        a = argmax(ActionDistribution(env.K , tq[n] , it))
                        #env.FitAck()
                    
                    env.take_action(n , a , prb)
                    
                    output += '    user: ' + str(n) + '\r\n'
                    output += '      State: ' + str(x[time]) + '\r\n'
                    output += '      Capacity: ' + str(env.capacity) + '\r\n'
                    output += '      Q: ' + str(tq[n]) + '\r\n'
                    output += '      Action Distribution: ' + str(ActionDistribution(env.K , tq[n] , it)) + '\r\n'
                    output += '      Chosen action: ' + str(a) + '\r\n\r\n'
                    ### Last
                    if n == env.N - 1:
                        NextState = []
                        env.FitAck()
                        reward = 0
                        rwrd = 0
                        for v in range(N):    
                            env.reward(v , it)
                            rwrd += env.r[v]
                        
                        for i in range(env.N):
                            
                            NextState.append([])
                            oo = copy(env.observe(i))
                            NextState[i] = oo
                            if scheme == "competitive":
                                for j in range(env.N):
                                    if env.AggregatedReward[i] != 0:
                                        reward += copy(env.r[i]/env.AggregatedReward[i])
                                    else:
                                        reward += env.r[i]

                            if scheme == "sum rate":
                                if t == T-1:
                                    for nn in range(env.N):
                                        reward += copy(env.AggregatedReward[nn])
                                    # or reward = f(env.AggregatedReward[n] , 0)
                            if scheme == "log-sum rate":
                                if t == T-1:
                                    for nn in range(env.N):
                                        reward += fair(copy(env.AggregatedReward[nn]) , 1)
                                else:
                                    reward = 0
                            NextState[i] = (NextState[i])[newaxis]
                            Q1 = DQN1.predict(NextState[i])
                            Q2 = DQN2.predict(NextState[i])
                            aa = copy(env.PreAction(i))
                            output += ' reward: ' + str(reward) + '\r\n'
                            tq[i][aa] = reward + Q2[0][argmax(Q1[0])]
                            Q[time - (N-1) + i] = tq[i]
                            output += ' Q2[0][argmax(Q1[0])]: ' + str(Q2[0][argmax(Q1[0])]) + "\r\n"
                            #output += 'Q[time]: ' + str(Q[2]) + '\r\n'
                            output += ' Q[' + str(time - (N-1) + i) + ']: ' + str(Q[time - (N-1) + i]) + '\r\n\r\n'
                            
                            if env.state[i][-1] > 0:
                                throughput += (1)
                            data[t].append(aa)
                
                #in T
            
            #in M
            dd =[]
            throughput /= (T*K)

            #figure
            if it >= iteration-10 or it%100==0:
                if throughput >= maxx and scheme != 'uniform':
                    maxx = throughput
                    fig = plt.figure(episode , figsize=(20,5))
                    dd = data[-1]
                    data = list(map(list, zip(*data)))
                    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                                     nrows_ncols=(1 , 1),  # creates 2x2 grid of axes
                                     axes_pad=0.1,  # pad between axes in inch.
                                     )
                    grid[0].imshow(data)  # The AxesGrid object work as a list of axes.
                    plt.savefig(scheme + " " + str(episode + 1) + ".png" , bbox_inches="tight")
                     
        #in iteration
        if it%100==0:
            f = open("Qs.txt" , "w+")
            f.write(output)
            f.close()
            
        if it == 3:
            f = open("Qs3.txt" , "w+")
            f.write(str(Q) + '\r\n')
            for b in range(len(Q)):
                f.write(str(x[b]) + str(Q[b]) + "\r\n")
            f.close()
            
        #fitting
        DQN1.fit(x , Q , epochs=1, verbose = 0)
        if it % 5 == 0:
            DQN2.set_weights(DQN1.get_weights())
            
        if dd != []:
            print(dd)
        if throughput >= .5:
            excc = '!!!'
        else:
            excc = ''
            
        #MA
        MA += throughput
        if it%10==0:
            MA/=10
            excc += "% 12.2f" % (100*MA)
            MA = 0
            
        print('throughput for iteration %i : ' % (it + 1) , "% 12.2f" % (100*throughput) , excc)
        
    
    f = open("qs" + " of " + scheme + ".txt" , "w+")
    f.write(output)
    f.close()
    return env.AggregatedReward , throughput , Q[-1] , DQN1

def ActionDistribution(K , Q , it , *con):
    alpha = .05 - .05*it/(iteration-1)
    beta = 19*it/(iteration-1) + 1
    s = 0
    p = []
    for a in range(K+1):
        try:
            s += exp(beta*Q[a])
        except OverflowError:
            s += float('inf')
        
    for a in range(K + 1):
        try:
            p.append( (((1 - alpha) * exp(beta * Q[a])) / s) + alpha / (K + 1) )
        except OverflowError:
            p.append(float('inf'))
    return p

def fair(x , alpha):
    ep = .000000000001
    if x <= 0:
        x = ep
    #return pow(x , 1 - alpha) / (1 - alpha)
    return log(x , 10)

def experience(env , DQN1):
    env.reset()
    data = []
    throughput = 0
    for t in range(100):
        data.append([])
        data[t] = []
        for n in range(N):
            x = copy(env.observe(n))
            Q = DQN1.predict(x[newaxis])
            Q = Q[0]
            a = argmax(ActionDistribution(env.K , Q , iteration-1 , True))
            env.take_action(n , a , prb)
            data[t].append(a)
            
            if n == env.N - 1:
                env.FitAck()
                
                for i in range(N):
                    if env.state[i][-1] > 0:
                        throughput += 1

    throughput /= (100*K)

    return throughput , data

                


#Body
env = Env(N , K)
r , t , q , dqn = iterate(M , T , "sum rate" , env , prb)

while True:
    FigNum = 0
    s = input("Exit?[y/n]: ")
    if s=='y':
        break
    
    thr = []
    data = []
    for it in range(100):
        thr.append(0)
        data.append([])
        thr[it] , data[it] = experience(env , dqn)
    
    scheme = 'competitive'
    for i in range(len(thr)):
        FigNum += 1
        print('throughput at experience #' , i+1 , ' : ' , "% 12.2f" % (100*thr[i]))
        print(data[i][-1])
        #figure
        fig = plt.figure(FigNum , figsize=(20,5))
        data[i] = list(map(list, zip(*data[i])))
        grid = ImageGrid(fig, 111,                  # similar to subplot(111)
                            nrows_ncols=(1 , 1),    # creates 2x2 grid of axes
                            axes_pad=0.1,           # pad between axes in inch.
                            )
        grid[0].imshow(data[i])  # The AxesGrid object work as a list of axes.
        plt.savefig('rslt/' + scheme + " " + str(FigNum) + ".png" , bbox_inches="tight")
        
