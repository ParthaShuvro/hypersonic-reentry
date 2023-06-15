#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sys
sys.path.append('C:\\Users\\User')
from typing import Optional, Union
import numpy as np
from numpy import sin, cos, tan, exp
import gym
import pandas as pd
#import csv
#import matplotlib
#from matplotlib import pyplot as plt
from gym import spaces
#from gym.envs.classic_control import utils
#from gym.error import DependencyNotInstalled
#from gym.utils.renderer import Renderer
from gym.envs.classic_control.Planet import Planet



class HRVEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    """
    ### Description

    # Observation space

The observation is a `ndarray` with shape `(6,)` with the values

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Radial distance (km)  |     6380            | 6488              |
    | 1   | Longitude (°)         |     -180            |  180              |
    | 2   | Latitude (°)          |      -90            |   90              |
    | 3   | Velocity (m/s)        |     1000            | 6550              |
    | 4   | Flight path angle (°) |      -90            |   90              |
    | 5   | Heading angle (°)     |      -90            |   90              |

# Action space

The action is a `ndarray` with shape `(2,)` with the values

    | Num |     Action            | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Angle of attack (°)   |     0               | 30                |
    | 1   | Bank angle (°)        |   -90               | 90                |

# Transition dynamics

The transition dynamics being followed is according to Vinh(1981). Initially 3dof model is being studied and upon successful completion 6dof model will be applied.

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "single_rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, PlanetModel=Planet('Earth'), render_mode: Optional[str] = None):
        self.planet = PlanetModel
        self.mass = 907.2
        self.area = 0.4839
        self.error_old_alpha = 0
        self.error_old_sigma = 0
        self.integral_error_alpha = 0
        self.integral_error_sigma = 0
        self.kp = 0.5
        self.ki = 0.8
        self.kd = 0.4
        self.tau = 0.2  # seconds between state updates        
        self.kinematics_integrator = "euler"
       
        self.initial_state = np.array([6470.0e3, 0.0, 0.0, 7000.0, -0.0349, 0.7854], dtype=np.float32)
        self.final_state = np.array([6400.0e3, 1.2217, 0.6981, 1000.0, -0.0873, -0.7854], dtype=np.float32)#2.4435,2.0944,1.2217, 0.6981, 1.7454, 0.5236,1.3963, 0.8727,-0.0873, -0.7854
        
        self.alpha_min = 0.0873
        self.alpha_max = 0.5236
        self.alpha_range = self.alpha_max-self.alpha_min
        self.sigma_min = -np.pi/2
        self.sigma_max = np.pi/2
        self.alpha_old = 0.4363
        self.sigma_old = -0.5236
        self.sigma_range = self.sigma_max-self.sigma_min

        self.act_min = np.array([0.0873, -np.pi/2], dtype=np.float32) # min [alpha, sigma] 
        self.act_max = np.array([0.5236, np.pi/2], dtype=np.float32) # max [alpha, sigma]
        
        self.obs_min = np.array([6388.0e3, -np.pi, -np.pi/2, 999.0, -np.pi/2, -np.pi], dtype=np.float32) # min [r, theta, phi, v, gamma, psi]
        self.obs_max = np.array([6478.0e3, np.pi, np.pi/2, 7050.0, np.pi/2, np.pi], dtype=np.float32) #max [r, theta, phi, v, gamma, psi]
        
        #self.act_min = np.array([-1.0, -1.0], dtype=np.float32) # min [alpha, sigma] 
        #self.act_max = np.array([1.0, 1.0], dtype=np.float32) # max [alpha, sigma]
        
        self.action_space = spaces.Box(self.act_min, self.act_max, dtype=np.float32)
        self.observation_space = spaces.Box(self.obs_min, self.obs_max, dtype=np.float32)
        """
        # normalization
        self.act_min = np.array([-1.0, -1.0], dtype=np.float32) # min [alpha, sigma] 
        self.act_max = np.array([1.0, 1.0], dtype=np.float32) # max [alpha, sigma]
        self.obs_min = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32) # min [r, theta, phi, v, gamma, psi]
        self.obs_max = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32) #max [r, theta, phi, v, gamma, psi]
        
        self.initial_state_norm = np.array([0.8222, 0, 0, 0.9835, -0.0222, 0.2500])#[0.8222, 0, 0, 0.9835, -0.0222, 0.2500]
        
        self.r_min = 6388.0e3
        self.r_max = 6478.0e3
        self.r_range = self.r_max-self.r_min
        self.theta_min = -np.pi
        self.theta_max = np.pi
        self.theta_range = self.theta_max-self.theta_min
        self.phi_min = -np.pi/2
        self.phi_max = np.pi/2
        self.phi_range = self.phi_max-self.phi_min
        self.v_min = 999.0
        self.v_max = 7050
        self.v_range = self.v_max-self.v_min
        self.gamma_min = -np.pi/2
        self.gamma_max = np.pi/2
        self.gamma_range = self.gamma_max-self.gamma_min
        self.psi_min = -np.pi
        self.psi_max = np.pi
        self.psi_range = self.psi_max-self.psi_min
        self.alpha_min = 0.0873
        self.alpha_max = 0.5236
        self.alpha_range = self.alpha_max-self.alpha_min
        self.sigma_min = -np.pi/2
        self.sigma_max = np.pi/2
        self.sigma_range = self.sigma_max-self.sigma_min
        """        
        #self.render_mode = render_mode
        #self.renderer = Renderer(self.render_mode, self._render)

        #self.screen_width = 600
        #self.screen_height = 400
        #self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.state_normalized = None
        self.steps_beyond_done = None
        self.steps_beyond_terminated = None
        
        
    def gravity(self, r):
        """ Returns gravitational acceleration at a given planet radius based on quadratic model 
        
            For radius in meters, returns m/s**2
            For non-dimensional radius, returns non-dimensional gravity 
        """
        return (self.planet.mu / (r ** 2))
    
    def step(self, action):
        assert self.state is not None, "Call reset before using step method."
        
        action = self.act(action)
        alpha = action[0]
        sigma = action[1]
        
        """  
        if self.steps_beyond_terminated is None:
            #action = np.array([self.alpha_old, self.sigma_old])
            action = self.pid(action, 0)
        else:
            action = self.pid(action, int(self.steps_beyond_terminated))
            
        alpha = action[0]
        sigma = action[1]
        action = self.act(action)
        #alpha = action[0]
        #sigma = action[1]
        alpha_norm = action[0]
        sigma_norm = action[1]
        
        alpha = self.state_denorm(alpha_norm, self.alpha_min, self.alpha_range)
        sigma = self.state_denorm(sigma_norm, self.sigma_min, self.sigma_range)
        
        """ 
        r, theta, phi, v, gamma, psi = self.state

        
        """
        #normalized 
        
        assert self.state_normalized is not None, "Call reset before using step method."

        r_norm, theta_norm, phi_norm, v_norm, gamma_norm, psi_norm = self.state_normalized
        
        
        r = self.state_denorm(r_norm, self.r_min, self.r_range)
        theta = self.state_denorm(theta_norm, self.theta_min, self.theta_range)
        phi = self.state_denorm(phi_norm, self.phi_min, self.phi_range)
        v = self.state_denorm(v_norm, self.v_min, self.v_range)
        gamma = self.state_denorm(gamma_norm, self.gamma_min, self.gamma_range)
        psi = self.state_denorm(psi_norm, self.psi_min, self.psi_range)
        
        alpha_norm = action[0]
        sigma_norm = action[1]
        
        alpha = self.state_denorm(alpha_norm, self.alpha_min, self.alpha_range)
        sigma = self.state_denorm(sigma_norm, self.sigma_min, self.sigma_range)
        """
        m = self.mass
        g = self.gravity(r)
        omega = self.planet.omega
        h = (r - self.planet.radius)
        rho, a = self.planet.atmosphere(h)
        M = v / a
        K = 5.188e-8        
               
        
        [cl0, cl1, cl2, cl3] = [-0.2317, 0.0513, 0.2945, -0.1028]
        [cd0, cd1, cd2, cd3] = [0.024, 7.24e-4, 0.406, -0.323]
        
        cL = cl1 * (alpha * 180 / np.pi)   + cl2 * exp(cl3 * M) + cl0 # lift coefficient
        cD = cd1 * (alpha * 180 / np.pi) ** 2 + cd2 * exp(cd3 * M) + cd0 # drag coefficient
        
        #print(alpha, cL, cD)
        #aerodynamic heating
        qD = K * (rho ** 0.5) * (v ** 3.15)
        qD_max = 2e8
        #dynamic pressure
        pD = 0.5 * rho * v ** 2
        pD_max = 5e5
        
        L = pD * self.area * abs(cL) # lift force
        D = pD * self.area * cD # drag force
        #normal load
        nL = np.sqrt(L**2 + D**2) / (m * g)
        nL_max = 10
        
        
        # dynamics equation
         
        # Vinh
        
        dr = v * sin(gamma) # radial distance
        dtheta = (v * cos(gamma) * cos(psi)) / (r * cos(phi)) # longitude
        dphi = (v * cos(gamma) * sin(psi)) / r # latitude
        dv = (-D )/ m - g * sin(gamma) + (omega ** 2) * r * cos(phi) * (sin(gamma) * cos(phi) - cos(gamma) * sin(phi) * sin(psi)) # velocity relative to Earth
        dgamma = L * cos(sigma) / (m * v) + cos(gamma) * (((v ** 2) - (g * r))/(r * v)) + 2 * omega * cos(psi) * cos(phi) + (omega ** 2) * (r / v) * cos(phi) * (cos(phi) * cos(gamma) + sin(phi) * sin(psi) * sin(gamma))  #  flight path angle
        dpsi = (L * sin(sigma)) / (m * v * cos(gamma)) - (v / r) * cos(gamma) * cos(psi) * tan(phi) + 2 * omega * (tan(gamma) * cos(phi) * sin(psi) - sin(phi)) - ((omega ** 2) * r * sin(phi) * cos(phi) * cos(psi)) / (v * cos(gamma)) # heading angle
            
        
            
        if self.kinematics_integrator == "euler":
            r = r + self.tau * dr
            theta = theta + self.tau * dtheta
            phi = phi + self.tau * dphi
            v = v + self.tau * dv
            gamma = gamma + self.tau * dgamma
            psi = psi + self.tau * dpsi
            
        if float(psi)>np.pi:
            psi = -(2*np.pi-psi)
        if float(psi)<-np.pi:
            psi = (2*np.pi+psi)
        self.state = np.array([r, theta, phi, v, gamma, psi])
        
        terminated = bool(
            np.any(self.state < self.obs_min)
            or np.any(self.state > self.obs_max)
            or qD > qD_max
            or pD > pD_max 
            or nL > nL_max
            )
        
        """
        # normalized 
        r_norm = self.state_norm(r, self.r_min, self.r_range)
        theta_norm = self.state_norm(theta, self.theta_min, self.theta_range)
        phi_norm = self.state_norm(phi, self.phi_min, self.phi_range)
        v_norm = self.state_norm(v, self.v_min, self.v_range)
        gamma_norm = self.state_norm(gamma, self.gamma_min, self.gamma_range)
        psi_norm = self.state_norm(psi, self.psi_min, self.psi_range)
        
        self.state_normalized = np.array([r_norm, theta_norm, phi_norm, v_norm, gamma_norm, psi_norm])
        
        #print(h, cL, cD, v)
        terminated = bool(
            np.any(self.state_normalized < self.obs_min)#self.state_normalized
            or np.any(self.state_normalized > self.obs_max)#self.state_normalized
            or qD > qD_max
            or pD > pD_max 
            or nL > nL_max
        )
        """
        if self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 1
            reward = 0
        else:
            if not terminated:
                self.steps_beyond_terminated += 1
                
                reward = 4000 - (r-self.final_state[0])/70 - (v-self.final_state[3])/6 - \
                abs(theta-self.final_state[1])*600- abs(phi-self.final_state[2])*1500-\
                    nL*100 - pD/500
                if np.all(np.array([(r-self.final_state[0]), (v-self.final_state[3])]) <= np.array([5.0,5.0])):
                    reward += 100000
                if np.all(np.array([(r-self.final_state[0]), (v-self.final_state[3])]) <= np.array([5.0, 1.0])):
                    reward += 100000
                    terminated = True
                if np.all(abs(self.state - self.final_state) <= np.array([10.0, 0.20, 0.10, 5.0, 0.20, 0.50])):
                    reward += 1000000
                    terminated = True
                """               
                # TQC ->seed(5)
                
                reward = 4000 - (r-self.final_state[0])/70 - (v-self.final_state[3])/6 - \
                abs(theta-self.final_state[1])*600- abs(phi-self.final_state[2])*1500-\
                    nL*100 - pD/500
                
                # DDPG ->seed(3)
                reward = 4000 - (r-self.final_state[0])/70 - (v-self.final_state[3])/6 - \
                abs(theta-self.final_state[1])*500- abs(phi-self.final_state[2])*1500-\
                    nL*100 - pD/500
                    
                
                    
                reward = 45000000 - (r-self.final_state[0])*500 - (v-self.final_state[3])*1500 - \
                    abs(theta-self.final_state[1])*1000 -abs(phi-self.final_state[2])*1000 -\
                        nL*5000 - pD/20
                        
                
                # TD3 normalized ->seed(3)
                reward = 10000 - (r-self.final_state[0])/35 - (v-self.final_state[3])/3 - \
                abs(theta-self.final_state[1])*1000- abs(phi-self.final_state[2])*2000-\
                    nL*500 - pD/50
                    
                # TD3
                reward = 15000 - (r-self.final_state[0])/14 - (v-self.final_state[3]) - \
                abs(theta-self.final_state[1])*1000- abs(phi-self.final_state[2])*1500-\
                    nL*1800 - pD/50
                 #****   
                reward = 15000 - (r-self.final_state[0])/14 - (v-self.final_state[3]) - \
                abs(theta-self.final_state[1])*1000- abs(phi-self.final_state[2])*1500-\
                    nL*1600 - pD/50
                """
                
                if dr<0:
                    
                    if np.any(abs(self.state - self.final_state) >= np.array([70.0e4, 1.30, 0.7, 6010.0, 2.5, 2.5])):
                        reward += -500                 
                    elif np.all(abs(self.state - self.final_state) <= np.array([15.0, 0.05, 0.05, 10.0, 0.10, 0.50])):
                        reward += 150000  
                    elif np.all(abs(self.state - self.final_state) <= np.array([20.0, 0.10, 0.10, 25.0, 0.20, 0.50])):
                        reward += 100000
                    elif np.all(abs(self.state - self.final_state) <= np.array([50.0, 0.10, 0.10, 50.0, 0.30, 0.50])):
                        reward += 50000
                    elif np.all(abs(self.state - self.final_state) <= np.array([100.0, 0.10, 0.10, 75.0, 0.35, 0.50])):
                        reward += 40000
                    elif np.all(abs(self.state - self.final_state) <= np.array([150.0, 0.10, 0.10, 100.0, 0.40, 0.50])):
                        reward += 35000
                    elif np.all(abs(self.state - self.final_state) <= np.array([200.0, 0.10, 0.10, 200.0, 0.50, 0.50])):
                        reward += 30000
                    elif np.all(abs(self.state - self.final_state) <= np.array([250.0, 0.12, 0.12, 300.0, 0.50, 0.55])):
                        reward += 25000
                    elif np.all(abs(self.state - self.final_state) <= np.array([300.0, 0.14, 0.14, 400.0, 0.50, 0.55])):
                        reward += 20000
                    elif np.all(abs(self.state - self.final_state) <= np.array([350.0, 0.16, 0.16, 500.0, 0.60, 0.60])):
                        reward += 15000
                    elif np.all(abs(self.state - self.final_state) <= np.array([400.0, 0.18, 0.18, 600.0, 0.60, 0.60])):
                        reward += 10000
                    elif np.all(abs(self.state - self.final_state) <= np.array([500.0, 0.20, 0.20, 700.0, 0.60, 0.65])):
                        reward += 8000
                    elif np.all(abs(self.state - self.final_state) <= np.array([800.0, 0.22, 0.22, 800.0, 0.60, 0.65])):
                        reward += 7000
                    elif np.all(abs(self.state - self.final_state) <= np.array([1000.0, 0.25, 0.25, 900.0, 0.60, 0.70])):
                        reward += 6000
                    elif np.all(abs(self.state - self.final_state) <= np.array([2000.0, 0.30, 0.30, 1000.0, 0.60, 0.70])):
                        reward += 5000
                        
                        #if r>6398.0e3:
                            #reward += 100000
                        #else: 
                            #reward += 1000
                        
                            
                    elif np.all(abs(self.state - self.final_state) <= np.array([2500.0, 0.35, 0.30, 1500.0, 0.60, 0.80])):
                        if float(r)>float(6400.0e3):
                            reward += 3000
                        
                        #if r>6398.0e3:
                            #reward += 70000
                        
                    elif np.all(abs(self.state - self.final_state) <= np.array([3000.0, 0.40, 0.30, 2000.0, 0.60, 0.80])):                    
                        if float(r)>float(6400.0e3):
                            reward += 2500
                        
                        #if r>6398.0e3:
                            #reward += 40000
                       
                    elif np.all(abs(self.state - self.final_state) <= np.array([4000.0, 0.45, 0.30, 2500.0, 0.60, 0.85])):
                        #reward += 26000
                        if float(r)>float(6400.0e3):
                            reward += 2200
                    elif np.all(abs(self.state - self.final_state) <= np.array([5000.0, 0.50, 0.35, 2700.0, 0.60, 0.85])):
                        #reward += 23000                       
                        if float(r)>float(6400.0e3):
                            reward += 2100
                    elif np.all(abs(self.state - self.final_state) <= np.array([6000.0, 0.55, 0.35, 2900.0, 0.60, 0.90])):                   
                        #reward += 20000
                        if float(r)>float(6400.0e3):
                            reward += 2000
                    elif np.all(abs(self.state - self.final_state) <= np.array([7000.0, 0.60, 0.35, 3100.0, 0.60, 0.90])):
                        #reward += 19000
                        if float(r)>float(6400.0e3):
                            reward += 1900
                    elif np.all(abs(self.state - self.final_state) <= np.array([8000.0, 0.65, 0.40, 3300.0, 0.60, 1.0])):
                        #reward += 18000
                        if float(r)>float(6400.0e3):
                            reward += 1800
                    elif np.all(abs(self.state - self.final_state) <= np.array([10.0e3, 0.70, 0.40, 3500.0, 0.60, 1.5])):
                        #reward += 17000
                        if float(r)>float(6400.0e3):
                            reward += 1700
                    elif np.all(abs(self.state - self.final_state) <= np.array([12.0e3, 0.75, 0.40, 3700.0, 1.0, 2.0])):
                        reward += 1600
                    elif np.all(abs(self.state - self.final_state) <= np.array([14.0e3, 0.75, 0.40, 3900.0, 1.0, 2.0])):
                        reward += 1500
                    elif np.all(abs(self.state - self.final_state) <= np.array([16.0e3, 0.80, 0.40, 4100.0, 1.0, 2.0])):
                        reward += 1400
                    elif np.all(abs(self.state - self.final_state) <= np.array([18.0e3, 0.80, 0.40, 4300.0, 1.0, 2.0])):
                        reward += 1300
                    elif np.all(abs(self.state - self.final_state) <= np.array([20.0e3, 0.85, 0.40, 4500.0, 1.0, 2.0])):
                        reward += 1200            
                    elif np.all(abs(self.state - self.final_state) <= np.array([22.0e3, 0.85, 0.40, 4600.0, 1.0, 2.0])):
                        reward += 1100
                    elif np.all(abs(self.state - self.final_state) <= np.array([23.0e3, 0.90, 0.45, 4650.0, 1.0, 2.0])):
                        reward += 1000
                    elif np.all(abs(self.state - self.final_state) <= np.array([24.0e3, 0.90, 0.45, 4700.0, 1.0, 2.0])):
                        reward += 900
                    elif np.all(abs(self.state - self.final_state) <= np.array([25.5e3, 0.95, 0.45, 4750.0, 1.0, 2.0])):
                        reward += 800
                    elif np.all(abs(self.state - self.final_state) <= np.array([26.0e3, 0.95, 0.45, 4800.0, 2.0, 2.0])):
                        reward += 700
                    elif np.all(abs(self.state - self.final_state) <= np.array([27.0e3, 1.00, 0.45, 4850.0, 2.0, 2.0])):
                        reward += 600
                    elif np.all(abs(self.state - self.final_state) <= np.array([28.0e3, 1.00, 0.45, 4900.0, 2.0, 2.0])):
                        reward += 500
                    elif np.all(abs(self.state - self.final_state) <= np.array([29.0e3, 1.05, 0.45, 4950.0, 2.0, 2.0])):
                        reward += 400
                    elif np.all(abs(self.state - self.final_state) <= np.array([30.0e3, 1.05, 0.50, 5000.0, 2.0, 2.0])):
                        reward += 300
                    elif np.all(abs(self.state - self.final_state) <= np.array([32.0e3, 1.05, 0.50, 5050.0, 2.0, 2.0])):
                        reward += 200
                    elif np.all(abs(self.state - self.final_state) <= np.array([34.0e3, 1.10, 0.50, 5100.0, 2.0, 2.0])):
                        reward += 150
                    elif np.all(abs(self.state - self.final_state) <= np.array([36.0e3, 1.10, 0.50, 5150.0, 2.0, 2.0])):
                        reward += 140
                    elif np.all(abs(self.state - self.final_state) <= np.array([38.0e3, 1.10, 0.50, 5200.0, 2.0, 2.0])):
                        reward += 120
                    elif np.all(abs(self.state - self.final_state) <= np.array([40.0e3, 1.15, 0.50, 5250.0, 2.0, 2.0])):
                        reward += 100
                    elif np.all(abs(self.state - self.final_state) <= np.array([42.0e3, 1.15, 0.50, 5300.0, 2.0, 2.0])):
                        reward += 90
                    elif np.all(abs(self.state - self.final_state) <= np.array([44.0e3, 1.15, 0.50, 5350.0, 2.0, 2.0])):
                        reward += 80
                    elif np.all(abs(self.state - self.final_state) <= np.array([46.0e3, 1.20, 0.55, 5400.0, 2.0, 2.0])):
                        reward += 70
                    elif np.all(abs(self.state - self.final_state) <= np.array([48.0e3, 1.20, 0.55, 5450.0, 2.0, 2.0])):
                        reward += 60
                    elif np.all(abs(self.state - self.final_state) <= np.array([50.0e3, 1.20, 0.55, 5500.0, 2.0, 2.0])):
                        reward += 50
                    elif np.all(abs(self.state - self.final_state) <= np.array([52.0e3, 1.20, 0.55, 5550.0, 2.0, 2.0])):
                        reward += 40
                    elif np.all(abs(self.state - self.final_state) <= np.array([54.0e3, 1.25, 0.55, 5600.0, 2.0, 2.0])):
                        reward += 35
                    elif np.all(abs(self.state - self.final_state) <= np.array([56.0e3, 1.25, 0.60, 5650.0, 2.0, 2.0])):
                        reward += 30
                    elif np.all(abs(self.state - self.final_state) <= np.array([58.0e3, 1.25, 0.60, 5700.0, 2.0, 2.0])):
                        reward += 25
                    elif np.all(abs(self.state - self.final_state) <= np.array([60.0e3, 1.25, 0.60, 5750.0, 2.0, 2.0])):
                        reward += 20
                    elif np.all(abs(self.state - self.final_state) <= np.array([62.0e3, 1.25, 0.65, 5800.0, 2.0, 2.0])):
                        reward += 15
                    elif np.all(abs(self.state - self.final_state) <= np.array([64.0e3, 1.25, 0.60, 5850.0, 2.0, 2.0])):
                        reward += 12
                    elif np.all(abs(self.state - self.final_state) <= np.array([66.0e3, 1.30, 0.70, 5900.0, 2.0, 2.0])):
                        reward += 10
                    elif np.all(abs(self.state - self.final_state) <= np.array([68.0e3, 1.30, 0.70, 5950.0, 2.0, 2.0])):
                        reward += 8
                    elif np.all(abs(self.state - self.final_state) <= np.array([70.0e3, 1.30, 0.70, 6000.0, 2.0, 2.0])):
                        reward += 5
                   
                else:
                    if np.all(abs(self.state - self.final_state) <= np.array([2500.0, 0.60, 0.40, 850.0, 0.5, 0.5])):
                        if float(r)<float(6400.0e3):
                            reward += 2400
                    elif np.all(abs(self.state - self.final_state) <= np.array([3000.0, 0.65, 0.40, 1000.0, 0.5, 0.5])):                    
                        if float(r)<float(6400.0e3):
                            reward += 2200
                    elif np.all(abs(self.state - self.final_state) <= np.array([4000.0, 0.70, 0.50, 1250.0, 0.5, 0.5])):
                        #reward += 26000
                        if float(r)<float(6400.0e3):
                            reward += 2000
                    elif np.all(abs(self.state - self.final_state) <= np.array([5000.0, 0.75, 0.50, 1500.0, 0.5, 0.6])):
                        #reward += 23000                       
                        if float(r)<float(6400.0e3):
                            reward += 1800
                    elif np.all(abs(self.state - self.final_state) <= np.array([6000.0, 0.80, 0.60, 1750.0, 0.5, 0.6])):                   
                        #reward += 20000
                        if float(r)<float(6400.0e3):
                            reward += 1600
                    elif np.all(abs(self.state - self.final_state) <= np.array([7000.0, 0.85, 0.60, 2000.0, 0.5, 0.7])):
                        #reward += 19000
                        if float(r)<float(6400.0e3):
                            reward += 1400
                    elif np.all(abs(self.state - self.final_state) <= np.array([8000.0, 0.90, 0.70, 2250.0, 0.5, 0.7])):
                        #reward += 18000
                        if float(r)<float(6400.0e3):
                            reward += 1200
                    elif np.all(abs(self.state - self.final_state) <= np.array([10.0e3, 0.95, 0.70, 2500.0, 0.5, 0.8])):
                        #reward += 17000
                        if float(r)<float(6400.0e3):
                            reward += 1000

            
            else:
                reward = -200000
                self.steps_beyond_terminated += 1
        

        df_action = pd.DataFrame(np.array([[alpha, sigma, self.steps_beyond_terminated]]))
        df_action.to_csv('D:\\project\\results & sims\\tqc-4\\data_action-tqc-4.csv', mode='a', index=False, header=False)
        df2 = pd.DataFrame(np.array([[r, theta, phi, v, gamma, psi, self.steps_beyond_terminated]]))
        df2.to_csv('D:\\project\\results & sims\\tqc-4\\data_obs-tqc-4.csv', mode='a', index=False, header=False)
        df3 = pd.DataFrame(np.array([[qD, pD, nL, self.steps_beyond_terminated]]))
        df3.to_csv('D:\\project\\results & sims\\tqc-4\\data_constraints-tqc-4.csv', mode='a', index=False, header=False)

        return np.array([self.state], dtype=np.float32), reward, terminated, False, {}
        
    # normalization 
        #return np.array([self.state_normalized], dtype=np.float32), reward, terminated, False, {}

    
    def act(self, action):
        # clipped action rate of change
        action = np.clip(action, np.array([self.alpha_old-0.0175, self.sigma_old-0.1047]), \
                         np.array([self.alpha_old+0.0175, self.sigma_old+0.1047])).astype(np.float32)
        a = action[0]
        s = action[1]
        # clipped action between min-max
        alpha_clip = np.clip(a, self.alpha_min, self.alpha_max)
        sigma_clip = np.clip(s, self.sigma_min, self.sigma_max)
        
        self.alpha_old = alpha_clip
        self.sigma_old = sigma_clip
        #alpha_norm = self.state_norm(alpha_clip, self.alpha_min, self.alpha_range)
        #sigma_norm = self.state_norm(sigma_clip, self.sigma_min, self.sigma_range)
        
        return np.array([alpha_clip, sigma_clip])
    """
    
    def act(self, action):
        #sample normalized action
       
        action = np.clip(action, self.act_min, self.act_max).astype(np.float32)
        alpha_norm = action[0]
        sigma_norm = action[1]
        
        # denormalize
        alpha = self.state_denorm(alpha_norm, self.alpha_min, self.alpha_range)
        sigma = self.state_denorm(sigma_norm, self.sigma_min, self.sigma_range)
        
        # constraints validation        
        if (alpha-self.alpha_old) > 0.0175:
            alpha_c = self.alpha_old + 0.0175
        elif (alpha-self.alpha_old) < -0.0175:
            alpha_c = self.alpha_old - 0.0175
        else:
            alpha_c = alpha
            
        if (sigma-self.sigma_old) > 0.1047:
            sigma_c = self.sigma_old + 0.1047
        elif (sigma-self.sigma_old) < -0.1047:
            sigma_c = self.sigma_old - 0.1047
        else:
            sigma_c = sigma
            
            
        self.alpha_old = alpha_c
        self.sigma_old = sigma_c
        
        
        alpha_norm = self.state_norm(alpha_c, self.alpha_min, self.alpha_range)
        sigma_norm = self.state_norm(sigma_c, self.sigma_min, self.sigma_range)
        
        return np.array([alpha_norm, sigma_norm])
    """
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.initial_state
        #self.state_normalized = self.initial_state_norm #normalized 
        self.steps_beyond_done = None
        self.steps_beyond_terminated = None
        self.alpha_old = 0.4363
        self.sigma_old = -0.5236
        
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {} 
        
    
    def state_norm(self, s, s_min, r):    
        # normalize between -1 to 1        
        s_n = 2*(s-s_min)/r - 1            
        return s_n
    
    def state_denorm(self, s_n, s_min, r):
        s = 1/2*(r*s_n + 2*s_min + r)
        return s
        
    def pid(self, action, i):
        action = self.act(action)
        """
        alpha = action[0]
        sigma = action[1]
        """
        alpha_n = action[0]
        sigma_n = action[1]
        
        alpha = self.state_denorm(alpha_n, self.alpha_min, self.alpha_range)
        sigma = self.state_denorm(sigma_n, self.sigma_min, self.sigma_range)
        
        data = pd.read_csv("D:\\project\\results & sim\\tqc-2\\data_action-tqc-2.csv")
        df = pd.DataFrame(data[531134:539843], columns=["alpha", "sigma", "steps"])
        
        alpha_r = df.at[i+531134,'alpha']
        sigma_r = df.at[i+531134,'sigma']
        
        error_alpha = alpha_r - alpha
        error_sigma = sigma_r - sigma
        #print(alpha_r, alpha, sigma, self.integral_error_alpha, self.error_old_alpha)
        d_error_alpha = (error_alpha - self.error_old_alpha) / self.tau
        d_error_sigma = (error_sigma - self.error_old_sigma) / self.tau
    
        self.integral_error_alpha += error_alpha * self.tau
        self.integral_error_sigma += error_sigma * self.tau
        
        alpha_gain = self.kp*error_alpha + self.ki*self.integral_error_alpha \
            + self.kd*d_error_alpha
        sigma_gain = self.kp*error_sigma + self.ki*self.integral_error_sigma \
            + self.kd*d_error_sigma
        
        alpha_new = alpha_r*(alpha_gain*alpha)/(1+alpha_gain*alpha)
        sigma_new = sigma_r*(sigma_gain*sigma)/(1+sigma_gain*sigma)
        self.error_old_alpha = error_alpha
        self.error_old_sigma = error_sigma
        
        return np.array([alpha_new, sigma_new])
        
        """
        # normalized
        if not return_info:
            return np.array(self.state_normalized, dtype=np.float32)
        else:
            return np.array(self.state_normalized, dtype=np.float32), {}   
       
    
       """
    """
    def state_norm(self, s, s_min, r):    
        # normalize between -1 to 1        
        s_n = 2*(s-s_min)/r - 1            
        return s_n
    
    def state_denorm(self, s_n, s_min, r):
        s = 1/2*(r*s_n + 2*s_min + r)
        return s
    
    def pid(self, action, i):
        action = self.act(action)
        alpha_n = action[0]
        sigma_n = action[1]
        
        alpha = self.state_denorm(alpha_n, self.alpha_min, self.alpha_range)
        sigma = self.state_denorm(sigma_n, self.sigma_min, self.sigma_range)
        
        data = pd.read_csv("D:\\project\\results & sim\\tqc-2\\data_action-2.csv")
        df = pd.DataFrame(data[531136:539844], columns=["alpha", "sigma", "steps"])
        
        
        alpha_r = df.at[i,'alpha']
        sigma_r = df.at[i,'sigma']
        
        error_alpha = alpha_r - alpha
        error_sigma = sigma_r - sigma
        print(alpha_r, alpha, sigma, self.integral_error_alpha, self.error_old_alpha)
        d_error_alpha = (error_alpha - self.error_old_alpha) / self.tau
        d_error_sigma = (error_sigma - self.error_old_sigma) / self.tau
    
        self.integral_error_alpha += error_alpha * self.tau
        self.integral_error_sigma += error_sigma * self.tau
        
        alpha_gain = self.kp*error_alpha + self.ki*self.integral_error_alpha \
            + self.kd*d_error_alpha
        sigma_gain = self.kp*error_sigma + self.ki*self.integral_error_sigma \
            + self.kd*d_error_sigma
        
        alpha_new = alpha_r*(alpha_gain*alpha)/(1+alpha_gain*alpha)
        sigma_new = sigma_r*(sigma_gain*sigma)/(1+sigma_gain*sigma)
        self.error_old_alpha = error_alpha
        self.error_old_sigma = error_sigma
        
        return np.array([alpha_new, sigma_new])
    """ 
    """  
   
    def act(self, action):
        action = np.clip(action, self.act_min, self.act_max).astype(np.float32)
        #print(action, action[0], action[1], self.alpha_old, self.sigma_old, (action[0]-self.alpha_old))
        
        if (action[0]-self.alpha_old) > 0.0175:
            alpha = self.alpha_old + 0.0175
        elif (action[0]-self.alpha_old) < -0.0175:
            alpha = self.alpha_old - 0.0175
        else:
            alpha = action[0]
            
        if (action[1]-self.sigma_old) > 0.1047:
            sigma = self.sigma_old + 0.1047
        elif (action[1]-self.sigma_old) < -0.1047:
            sigma = self.sigma_old - 0.1047
        else:
            sigma = action[1]
            
            
        self.alpha_old = alpha
        self.sigma_old = sigma
        
        #alpha = action[0]
        #sigma = action[1]
        return np.array([alpha, sigma])
    
class PID():
    def __init__(self):
        self.error_old_alpha = 0
        self.error_old_sigma = 0
        self.integral_error_alpha = 0
        self.integral_error_sigma = 0
        self.dt = 0.2
        self.kp = 1
        self.ki = 2
        self.kd = 3
    def compute(self, action, i):
        action = HRVEnv().act(action)
        alpha = action[0]
        sigma = action[1]
        
        data = pd.read_csv("D:\\project\\results & sim\\ppo-l12\\data_action-l125.csv")
        df = pd.DataFrame(data, columns=["alpha", "sigma", "steps"])
        
        
        alpha_r = df.at[i,'alpha']
        sigma_r = df.at[i,'sigma']
        
        error_alpha = alpha_r - alpha
        error_sigma = sigma_r - sigma
        print(alpha_r, alpha, self.integral_error_alpha, self.error_old_alpha)
        d_error_alpha = (error_alpha - self.error_old_alpha) / self.dt
        d_error_sigma = (error_sigma - self.error_old_sigma) / self.dt
    
        self.integral_error_alpha += error_alpha * self.dt
        self.integral_error_sigma += error_sigma * self.dt
        
        alpha_new = self.kp*error_alpha + self.ki*self.integral_error_alpha \
            + self.kd*d_error_alpha
        sigma_new = self.kp*error_sigma + self.ki*self.integral_error_sigma \
            + self.kd*d_error_sigma
            
        self.error_old_alpha = error_alpha
        self.error_old_sigma = error_sigma
        
        return np.array([alpha_new, sigma_new])
    
    
    
    
    """
# In[ ]:




