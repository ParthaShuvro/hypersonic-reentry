#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import random

import time
from DDPG.logger.result import result_log
#from result_plot import result_plot

class HrvEnv(Env):
    print_interval = 1
    def __init__(self, logger = None):
        self.env = Hrv()
        self.noiseRange = 1.0
        self.noiseMax = 1.0
        self.om = 0
        self.a = 0.6
        self.b = 0.4
        self.t = 0
        self.totStep = 0
        self.r = 0
        self.ep = 0
        if logger==None:
            self.perfs = result_log(algo="DDPG", l1=20, l2=10)
        else:
            self.perfs = logger
        self.actif = True
        #self.plot = result_plot()
    
    def state(self):
        return [self.env.getObservation()]
    def act(self, action):
        actNoise = action + self.noise_func()
        self.env.performAction(actNoise[0])
        r = self.env.getReward()
        self.t += 1
        self.r += r
        return actNoise, [r]
    def reset(self, noise=True):
        self.actif = True
        self.env.reset()
        self.om = 0
        self.totStep+=self.t
        if self.totStep != 0:
            self.perfs.addData(self.totStep, self.t, self.r)
        self.t = 0
        self.r = 0
        self.ep += 1
        if not noise:
            self.noiseRange = 0.0
        else:
            self.noiseRange = random.uniform(0.,self.noiseMax)
    def noise_func(self):
        self.om = self.om-self.a*self.om + self.b*random.gauss(0,1)*self.noiseRange
        return self.om
    def isFinished(self):
        if self.actif and not self.env.isFinished():
            return False
        else:
            self.actif = False
            return True
    def getActionSize(self):
        return 2
    def getStateSize(self):
        return 6
    def getActionBounds(self):
        return [[1.2], [-1.2]
               [-90], [90]]
    def printEpisode(self):
        print time.strftime("[%H:%M:%S]"), " Episode : " , self.ep, " steps : ", self.t, " reward : ", self.r, "noise : ", self.noiseRange
    def performances(self):
        pass#self.plot.clear()
        #self.plot.add_row(self.perfs)

