#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pybrain


# In[7]:


import sys
sys.path.append('C:\\Users\\User')
from scipy import cos
from pybrain.rl.environments.episodic import EpisodicTask
import Entry_Vehicle
import Entry
import Planet

import random
import math


class Control(EpisodicTask):

    # number of steps of the current trial
    steps = 0

    # number of the current episode
    episode = 0
    
    maxSteps = 99
    
    resetOnSuccess = True

    def __init__(self):
        self.reset()
        self.cumreward = 0

    def reset(self):
        self.state = self.GetInitialState()
        alpha = random.uniform(-20, 20)
        sigma = 0
    
    def getObservation(self):    
        self.state = self.Entry.__entry_3dof(self, s, alpha, sigma, t)
        return [(self.state)]
        
    def performAction(self, action):
        if self.done > 0:
            self.done += 1            
        else:
            self.state = self.DoAction(action, self.state)
            self.r, self.done = self.GetReward(self.state, action)
            self.cumreward += self.r
            
    def getReward(self):
        return self.r    

    def GetInitialState(self):
        # Initialize state values
        self.StartEpisode()
        r = 6448
        theta = 0
        phi = 0
        v = 6500
        gamma = 0
        psi = 90
        
        return [r, theta, phi, v, gamma, psi]
  
    def FinalState(self):
        # set final target values
        r = 6408
        theta = 50
        phi = 0
        v = 1600
        gamma = 0
        psi = 90
        
        return [r, theta, phi, v, gamma, psi]
    
    def StartEpisode(self):
        self.steps = 0
        self.episode = self.episode + 1
        self.done = 0
        
    def isFinished(self):
        if self.done>=1 and self.resetOnSuccess:
            self.reset()
            return True
        else:
            return self.done>=3
    

    def GetReward(self, s, a):
        #todo: formulate reward function
        # satisfy constraints either here or in doaction
        r = 0
        f = 0
        
        if (self.s-self.FinalState) == np.zeros(6):
            r = 100
        elif np.absolute(self.s-self.FinalState) < np.array([10, 5, 5, 100, 5, 5]):
            r = 50
        elif np.absolute(self.s-self.FinalState) < np.array([20, 10, 10, 400, 10, 10]):
            r = 10
        elif np.absolute(self.s-self.FinalState) < np.array([30, 20, 20, 1000, 20, 20]):
            r= 2
        elif np.absolute(self.s-self.FinalState) < np.array([40, 30, 30, 2000, 30, 30]):
            r = 0 
        elif np.absolute(self.s-self.FinalState) > np.array([40, 30, 30, 2000, 30, 30]):
            r = -1
        elif np.absolute(self.s-self.FinalState) > np.array([45, 30, 30, 2500, 30, 30]):
            r = -5
        elif np.absolute(self.s-self.FinalState) > np.array([50, 45, 45, 3000, 45, 45]):
            r = -10
        if self.steps >= self.maxSteps:
            f = 5
   
        return r, f

    def DoAction(self, a, s):

        alpha = self.alpha
        sigma = self.sigma
        # todo: satisfy constraints
        self.steps = self.steps + 1
        s_new = s + self.Entry.__entry_3dof(s, alpha, sigma, t)

        
        return [s_new]
    
    def mach(self):
        M = self.Entry.__entry_3dof.M
        
        return M
    
    def alpha(self):
        #alpha = self.get_action[0]
        ''''
        if self.step=0:
            alpha = random.uniform(-20, 20)
        else:
            pass
            '''
        alpha_old = self.AlphaOld
        alpha_new = random.uniform(-20, 20)
        dalpha = alpha_new - alpha_old
        
        if dalpha<-5:
            alpha = alpha_old - 5
        elif dalpha>5:
            alpha = alpha_old + 5
        else:
            alpha = alpha_new
        return alpha
    
    def AlphaOld(self):
        alpha_old = self.alpha
        return alpha_old
    
    def sigma(self):
        #sigma = self.get_action[1]
        '''  
        if self.step=0:
            sigma = 0
        else:
            pass
        '''
        sigma_old = self.SigmaOld
        sigma_new = random.uniform(-90, 90)
        dsigma = sigma_new - sigma_old
        
        if dsigma<-15:
            sigma = sigma_old - 15
        elif dsigma>15:
            sigma = sigma_old + 15
        else:
            sigma = sigma_new
        return sigma
    
    def SigmaOld(self):
        sigma_old = self.sigma
        return sigma_old
    
    def contraints(self):
        V = self.Entry.__entry_3dof[3]
        K = 9.289e-9
        rho = self.Planet.atmosphere[0]
        Cd, Cl = self.EntryVehicle.aerodynamic_coefficients(M, alpha)
        D = 0.5 * rho * V **2 * S * Cd
        L = (Cl/Cd) * D
        #aerodynamic heating
        Qd = K * rho ** 0.5 * V ** 3
        #dynamic pressure
        P_d = 0.5 * rho * V ** 2
        #normal load
        n_L = math.sqrt(L**2+D**2)/(self.mass*self.gravity)
        return [[Qd, P_d, n_L]]


# In[ ]:




