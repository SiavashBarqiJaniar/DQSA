# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 19:30:36 2019

@author: Siavash Barqi
"""
import numpy as np

class Env(object):
    
    def __init__(self , N , K, scheme):
        self.state = np.ndarray(shape = (N , 2*K + 2) , dtype = int)
        self.AggregatedReward = []
        self.N = N
        self.K = K
        self.scheme = scheme
        for s in range(N):
            self.AggregatedReward.append(0)
            self.state[s][0] = 1
            for e in range(K):
                self.state[s][e + 1] = 0
                self.state[s][e + K + 1] = 1
            self.state[s][-1] = 0
    
    def reset(self):
        self.AggregatedReward = []
        for s in range(self.N):
            self.AggregatedReward.append(0)
            self.state[s][0] = 1
            for e in range(self.K):
                self.state[s][e + 1] = 0
                self.state[s][e + self.K + 1] = 1
            self.state[s][-1] = 0
        #ts.reset
        return self.state
    
    def step(self, action, done):
        K = self.K
        N = self.N
        cap = np.ones(K)
        PreState = self.state
        state = np.zeros(shape=(N,2*K+2))
        r = np.zeros(N)
        M = np.zeros(N)
        sum_log_reward = np.zeros(N)
        reward = np.zeros(N)
        
        """=================================================================
        #                 K+1          K         1    =    2K+2
        #       State: [0 ... K] [K+1 ... 2K] [2K+1]
        ================================================================="""
        
        #take action
        for n in range(N):
            #initializing
            for i in range(K + 1):
                state[n][i] = 0
            for i in range(K):
                state[n][K+1+i] = 1
            #action & capacity
            state[n][action[n]] = 1
            if action[n]!=0:
                cap[action[n]-1] -= 1
                
        #fitting capacity
        for n in range(N):
            for i in range(K):
                if cap[i] == 1:
                    state[n][K + 1 + i] = 1
                else:
                    state[n][K + 1 + i] = 0
        
        #ACK signal
        for n in range(N):
            if action[n] == 0: #no tx
                state[n][-1] = 0
            else:
                if cap[action[n]-1] == 0: #cap = 0
                    state[n][-1] = 1
                else: #collision
                    state[n][-1] = 0
            
            r[n] = state[n][-1] #reward
            self.AggregatedReward[n] += r[n]
            M[n] = self.AggregatedReward[n]
            if M[n]==0:
                M[n] = 0.2
            sum_log_reward[n] = r[n]/M[n]
            
        self.state = state
        
        #reward
        if self.scheme == 'competitive':
            reward = r
        elif self.scheme == 'sum rate':
            if done == True:
                reward = sum(r)*np.ones(N)
        elif self.scheme == 'sum-log rate':
            #raise Exception('Not defined yet!')
            if done == True:
                reward = sum(sum_log_reward)*np.ones(N)
        else:
            raise Exception('Not such scheme name exists!')
            
        return PreState, action, state, reward
    
    def observe(self , n):
        return self.state[n]
        

    def PreAction(self , n):
        for i in range(self.K + 1):
            if self.state[n][i] == 1:
                return i
    
    def fair(x , alpha):
        ep = .000000000001
        if x <= 0:
            x = ep
        #return pow(x , 1 - alpha) / (1 - alpha)
        return log(x , 10)
    
    
    
    






























