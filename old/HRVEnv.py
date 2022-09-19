#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sys
sys.path.append('C:\\Users\\User')
import math
from typing import Optional, Union

import numpy as np
from numpy import sin, cos, tan

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer
from EntryVehicle import EntryVehicle
from Planet import Planet


class HRVEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    ### Description

    # Observation space

The observation is a `ndarray` with shape `(6,)` with the values

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Radial distance (km)  |     6400            | 7000              |
    | 1   | Longitude (°)         |        0            |   90              |
    | 2   | Latitude (°)          |        0            |   90              |
    | 3   | Velocity (m/s)        |     1000            | 7000              |
    | 4   | Flight path angle (°) |        0            |   90              |
    | 5   | Heading angle (°)     |        0            |   90              |

# Action space

The action is a `ndarray` with shape `(2,)` with the values

    | Num |     Action            | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Angle of attack (°)   |   -30               | 30                |
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

    def __init__(self, PlanetModel=Planet('Earth'), VehicleModel=EntryVehicle(), render_mode: Optional[str] = None):
        self.planet = PlanetModel
        self.vehicle = VehicleModel
        self.drag_ratio = 1
        self.lift_ratio = 1
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.initial_state = np.array([6448, 0, 0, 6500, 0, 90], dtype=np.float32)
        self.final_state = np.array([6408, 50, 0, 1600, 0, 90], dtype=np.float32)
        self.dist_scale = self.planet.radius
        self.acc_scale = self.planet.mu/(self.dist_scale**2)
        self.time_scale = np.sqrt(self.dist_scale/self.acc_scale)
        self.vel_scale = np.sqrt(self.dist_scale*self.acc_scale)
        self.mass_scale = 1
       
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        act_min = np.array([-30, -90], dtype=np.float32) # min [alpha, sigma] 
        act_max = np.array([30, 90], dtype=np.float32) # max [alpha, sigma]
        obs_min = np.array([6400, 0, 0, 1000, 0, 0], dtype=np.float32) # min [r, theta, phi, v, gamma, psi]
        obs_max = np.array([7000, 90, 90, 7000, 90, 90], dtype=np.float32) #max [r, theta, phi, v, gamma, psi]

        self.action_space = spaces.Box(act_min, act_max, dtype=np.float32)
        self.observation_space = spaces.Box(obs_min, obs_max, dtype=np.float32)

        #self.render_mode = render_mode
        #self.renderer = Renderer(self.render_mode, self._render)

        #self.screen_width = 600
        #self.screen_height = 400
        #self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_done = None
        
    def gravity(self, r):
        """ Returns gravitational acceleration at a given planet radius based on quadratic model 
        
            For radius in meters, returns m/s**2
            For non-dimensional radius, returns non-dimensional gravity 
        """
        return self.planet.mu/(r*self.dist_scale)**2/self.acc_scale
    
    def action(self):
        assert self.action_space.contains(action)
        
        alpha = self.action[0]
        sigma = self.action[1]
        
        if (alpha-alpha_old)>5:
            alpha += 5
        elif (alpha-alpha_old)<-5:
            alpha -= 5
        
        alpha_old = alpha
        
        if (sigma-sigma_old)>15:
            sigma += 15
        elif (sigma-sigma_old)<-15:
            sigma -= 15
        
        sigma_old = sigma
        
        self.action = [alpha, sigma]
        return self.action

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        # assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        r, theta, phi, v, gamma, psi = self.state
        
        g = self.gravity(r)
        omega = self.planet.omega
        h = (r - self.planet.radius)/self.dist_scale
        rho, a = self.planet.atmosphere(h*self.dist_scale)
        M = v*self.vel_scale/a
        
        alpha = self.action[0]
        sigma = self.action[1]
        
        cD, cL = self.vehicle.aerodynamic_coefficients(M, alpha)
        m = self.vehicle.mass
        f = np.squeeze(0.5*rho*self.vehicle.area*v**2/m)*self.dist_scale  # vel_scale**2/acc_scale = dist_scale 
        L = f*cL*self.lift_ratio
        D = f*cD*self.drag_ratio        


        dr = v*sin(gamma)
        dtheta = v*cos(gamma)*cos(psi)/r*cos(phi)
        dphi = v*cos(gamma)*sin(psi)/r
        dv = -D/m - g*sin(gamma) + omega**2*r*cos(phi)*(sin(gamma)*cos(phi) - cos(gamma)*sin(phi)*sin(psi))
        dgamma = L*cos(sigma)/m*v + cos(gamma)*((v**2-g*r)/r*v) + 2*omega*sin(psi)*cos(phi) + omega**2*r*cos(phi)*(cos(phi)*cos(gamma)+sin(phi)*sin(psi)*sin(gamma)) 
        dpsi = L*sin(sigma)/m*v*cos(gamma) - v*cos(gamma)*cos(psi)*tan(phi)/r + 2*omega*(tan(gamma)*cos(phi)*sin(psi)-sin(phi)) - omega**2*r*sin(phi)*cos(phi)*cos(psi)/cos(gamma)

        if self.kinematics_integrator == "euler":
            r = r + self.tau * dr
            theta = theta + self.tau * dtheta
            phi = phi + self.tau * dphi
            v = v + self.tau * dv
            gamma = gamma + self.tau * dgamma
            psi = psi + self.tau * dpsi

        self.state = (r, theta, phi, v, gamma, psi)
        
        terminated = bool(self.state <= obs_min or self.state >= obs_max)
        
        reward = 0
        
        if (self.state-self.final_state) == np.zeros(6):
            reward = 100
        elif np.absolute(self.state-self.final_state) < np.array([10, 5, 5, 100, 5, 5]):
            reward = 50
        elif np.absolute(self.state-self.final_state) < np.array([20, 10, 10, 400, 10, 10]):
            reward = 10
        elif np.absolute(self.state-self.final_state) < np.array([30, 20, 20, 1000, 20, 20]):
            reward= 2
        elif np.absolute(self.state-self.final_state) < np.array([40, 30, 30, 2000, 30, 30]):
            reward = 0 
        elif np.absolute(self.state-self.final_state) > np.array([40, 30, 30, 2000, 30, 30]):
            reward = -1
        elif np.absolute(self.state-self.final_state) > np.array([45, 30, 30, 2500, 30, 30]):
            reward = -5
        elif np.absolute(self.state-self.final_state) > np.array([50, 45, 45, 3000, 45, 45]):
            reward = -10

   
        #self.renderer.render_step()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.state = self.initial_state
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}
    """    
    def constraints(self):
        V = self.state[3]
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
        """


# In[ ]:




