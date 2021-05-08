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
from keras.layers import Input, Embedding, LSTM, Dense, add, average, concatenate, subtract, Reshape
from keras.models import Model, load_model
from math import *
import plot as plt
from numpy.random import choice

#Parameters
N = 3
K = 2
T = 50
M = 16
epsilon = 0.05
prb = 1 - epsilon
R = 10100 #50000
gamma = 0.985

#Funcs
def iterate(M, T, env, prb, path):
    #DQN
    inp = Input(shape=(2*K+2,), dtype=float32)
    #emb = Embedding(input_dim = 2 , output_dim = 64)(inp)
    r = Reshape((1, -1))(inp)#, input_shape=(12,))
    lstm = LSTM(100 , return_sequences=False, activation='relu')(r)
    
    #1
    value = Dense(10 , activation='relu')(lstm)
    v = Dense(20 , activation='relu')(value)
    OutputLayer = Dense(K+1)(v)
    """a = Dense(10 , activation='relu')(lstm)
    #a = Dense(16 , activation='relu')(a)
    ad = []

    for i in range(K + 1):
        ad.append(Dense(1)(a))
        
        
    averaged = average(ad)
    
    advantage = concatenate(inputs = ad , axis=-1)
    
    subtracted = subtract([advantage , averaged])
    OutputLayer = add([value , subtracted])"""
    
    #2
    inp2 = Input(shape=(2*K+2,), dtype=float32)
    #emb = Embedding(input_dim = 2 , output_dim = 64)(inp)
    r2 = Reshape((1, -1))(inp2)#, input_shape=(12,))
    lstm2 = LSTM(100 , return_sequences=False, activation='relu')(r2)
    
    #1
    value2 = Dense(10 , activation='relu')(lstm2)
    v2 = Dense(20 , activation='relu')(value2)
    OutputLayer2 = Dense(K+1)(v2)
    """a2 = Dense(10 , activation='relu')(lstm2)
    #a2 = Dense(16 , activation='relu')(a2)
    ad2 = []

    for i in range(K + 1):
        ad2.append(Dense(1)(a2))
        
        
    averaged2 = average(ad2)
    
    advantage2 = concatenate(inputs = ad2 , axis=-1)
    
    subtracted2 = subtract([advantage2 , averaged2])
    OutputLayer2 = add([value2 , subtracted2])"""
    
    
    DQN1 = Model(inputs = inp , outputs = OutputLayer)
    DQN2 = Model(inputs = inp2 , outputs = OutputLayer2)
    
    #DQN1.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    #DQN2.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
    
    DQN1.compile(loss='mse', optimizer='adam')
    DQN2.compile(loss='mse', optimizer='adam')
    
    """
    print('Initialization')
    esw = array([1 , 0 , 0 , 0 , 0 , 0])[newaxis]
    dhg = array([0])[newaxis]
    dhgf = array([0,0,0])[newaxis]
    DQN1.fit(esw , dhgf , epochs=8, batch_size=4 , verbose = 0)
    print(DQN1.predict(esw))
    """
    print('START')
    
    N = env.N
    output = env.scheme
    excc = ""
    MA = 0
    maxx = 0
    throughput = 0
    prob = .95
    
    ### Loading DQN's
    #DQN1 = load_model('DQN1.h5')
    
    DQN2.set_weights(DQN1.get_weights())
    replayBufferx = ndarray(shape = (100, M*T*N, 2*K + 2))
    replayBufferQ = ndarray(shape = (100, M*T*N, K + 1))
    
    for it in range(R):
        print('throughput for iteration %i :' % (it))
        tq = ndarray(shape = (N , K + 1))
        OptA = ndarray(shape = (N , 2*K + 2))
        Q = ndarray(shape = (M*T*N , K + 1))
        x = ndarray(shape = (M*T*N , 2*K + 2) , dtype = int)
        act = []
        time = -1
        dd =[]

        for yy in range(15):
            output += '|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||\r\n'
        output += 'at iteration ' + str(it) + '\r\n'
        
        for episode in range(M):
            output += '\r\nEpisode : ' + str(episode) + '\r\n'
            throughput = 0
            data = []
            state = env.reset()
            
            for t in range(T):
                tprime = t
                output += '\r\n  TIME : ' + str(t) + '\r\n'
                data.append(t)
                data[t] = []
                a = np.zeros(N, dtype=int)

                for n in range(N):
                    time += 1
                    
                    """
                    if it < 256:
                        xx = copy(env.observe(n))
                        x[time] = copy(xx)
                        tq[n] = DQN1.predict((xx)[newaxis])
                        rand = np.random.randint(N)
                        #print(rand)
                        if rand%N <= env.K:
                            a = rand
                        else:
                            a = 0
                        """
                        
                    x[time] = copy(state[n])
                    x[time] = (x[time])[newaxis]
                    tq[n] = DQN1.predict((state[n])[newaxis])
                    outputtq = tq[n]
                    p , tq[n] = ActionDistribution(env.K , tq[n] , it, R, path)
                    
                    list_of_candidates = range(K + 1)
                    nn = sum(p)
                    probability_distribution = [x/nn for x in p]
                    number_of_items_to_pick = 1
                    
                    aMax = choice(list_of_candidates, number_of_items_to_pick,
                                  p=probability_distribution)
                    #aMax = argmax(p)
                    #aMax = argmax(tq[n])
                    
                    #prob = exp(-4.7 + 4.95*it/1000) # e^-4.7 =~ 0 e^-0.05 =~ 0.95
                    #prob = 1 - exp((0-it)/(1000/3)) # e^(-3) = .04
                    """"prob = 1 - .1*(.995**it)
                    prob = max(1 - 0.05, prob)
                    prob = prob - (prob%0.001)
                    """
                    #elif t < 5:
                    #    prob = 1 - .4
                    ### competitive: e^(-3)
                    ###              1 - .05
                    
                    ran = random.uniform(0.0 , 1.0)
                    if ran < prob:
                        a[n] = aMax # argmax(ActionDistribution(K , tq[n] , it))
                    else:
                        ran = random.randint(K) # random int between 0 and K-1
                        if ran < aMax:
                            a[n] = ran
                        else:
                            a[n] = ran + 1
                            
                    output += '    ' + str(state[n]) + ' --> Q --> ' + str(outputtq) + ' --> Dist --> ' + str(p) + '\r\n'
            
                ###=========================================================###
                ###                      in T
                ###=========================================================###
                
                output += '    Chosen action: ' + str(a) + '\r\n'
                
                done = False
                if t==T-1:
                    done = True
                _, action, state, reward = env.step(a, done)
                
                
                for i in range(N):
                    output += '    ' + str(state[i]) + '\r\n'
                output += '    reward: ' + str(reward) + '\r\n'
                
                for i in range(N):
                    s = (state[i])[newaxis]
                    Q1 = DQN1.predict(s)
                    Q2 = DQN2.predict(s)
                    tq[i][action[i]] = reward[i] + gamma*Q2[0][argmax(Q1[0])] #action[i]]
                    
                    Q[time - (N-1) + i] = tq[i]
                    if state[i][-1]==1:
                        throughput += (1)
                    data[t].append(action[i])
                    
                    act.append(action[i])
                    
                    output += '    Q2[' + str(action[i]) + ']: ' + str(Q2[0][action[i]])
                    output += ' --> Q[' + str(time - (N-1) + i) + ']: ' + str(Q[time - (N-1) + i]) + '\r\n'
                
            ###=========================================================###
            ###                        in M
            ###=========================================================###
            
            """if env.scheme == 'sum rate' or env.scheme == 'sum-log rate':
                for i in range(T*N):
                    Q[time - i] += reward[0]"""
                
            throughput /= (T*K)
            
            if throughput >= .5:
                excc = '!!!'
            else:
                excc = ''
            
            print("episode %i: " % (episode) , "% 12.2f" % (100*throughput) , excc)
        
            if throughput>maxx:
                DQN1.save(path + 'DQN1.h5')
                maxx = throughput
            
            #figure
            if it >= R-10 or it%100==0 or maxx == throughput or throughput >= 0.65:
                plt.show(data, throughput, it, episode, path)
                dd = data[-1]
            
            if dd != []:
                print(dd)
                dd = []
                
            #MA
            MA += throughput
                     
        ###===============================================================###
        ###                         in iteration
        ###===============================================================###
        
        """if it%100==0:
            f = open("Qs" + str(it) + ".txt" , "w+")
            f.write(output)
            f.close()
            output = ''
            
        if it == 3:
            f = open("Qs3.txt" , "w+")
            f.write(str(Q) + '\r\n')
            for b in range(len(Q)):
                f.write(str(x[b]) + str(Q[b]) + "\r\n")
            f.close()
        """
        
        #replayBufferx[it%100] = x
        #replayBufferQ[it%100] = Q
        
        f = open(path + "history" + " of " + env.scheme + ' ' + str(int(it/1000)) + ".txt" , "a")
        f.write(output)
        f.close()
        output = ''

        #fitting
        #if it >= 99:
            #x = replayBufferx[i]
            #Q = replayBufferQ[i]
        batchSize = M*T*N # len(x)
        DQN1.fit(x , Q , batch_size=batchSize , epochs=1 , verbose = 0) #batch_size=50(time) epochs=10000(iteration) policy=absolutely random
        if it % 5 == 0: # % 80
            DQN2.set_weights(DQN1.get_weights())

        MA /= M
        print("% 12.2f" % (100*MA))
        MA = 0
    
    #Saving
    DQN1.save(path + 'DQN1.h5')
    
    return env.AggregatedReward , throughput , Q[-1] , DQN1

def ActionDistribution(K , Q , it, R , path, *con):
    #alpha = exp(-2.9 - 2.1*it/R) # e^2.9 = 0.05
    #alpha = alpha - (alpha%0.001)
    if it < 2000:
        #alpha = .05*(.995**it)
        alpha = 0.05 - 0.05*it/(2000 - 1)
    #beta =exp(2.95 - 7.65*it/R) # e^2.95 =~ 19.1
    #beta = beta - (beta/.001)
    #beta = 20 - beta
        #beta = 1.005**it
        beta = 1 + 19*it/(2000 - 1)
    else:
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
        #elif check == False:
        s += exp(beta*Q[a])
        ### exp(710) = inf
        
    #if check == True:
    #    s = float("inf")
        
    for a in range(K + 1):
        """if check == True:
            if beta*Q[a] >= 709:
                p.append( ((1 - alpha)/count) + alpha/(K + 1) )
            else:
                p.append(alpha/(K + 1))
        else:"""
        p.append( (((1 - alpha) * exp(beta * Q[a])) / s) + alpha / (K + 1) )
            
    return p, Q

"""=====================================================================
###                           Body
====================================================================="""
    
env2 = Env(N, K, 'competitive')
path = 'results/competitive/64/'
r2 , t2 , q2 , dqn2 = iterate(M, T, env2, prb, path)
        
