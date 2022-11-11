#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sys
sys.path.append('C:\\Users\\User')
from typing import Optional, Union
import numpy as np
from numpy import sin, cos, tan, exp
import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
from gym.utils.renderer import Renderer
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
        self.drag_ratio = 1
        self.lift_ratio = 1
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.alpha_old = 0.0
        self.sigma_old = 0.0
        self.initial_state = np.array([6458.0e3, 0.0, 0.0, 6500.0, -0.5236, 0], dtype=np.float32)
        self.final_state = np.array([6408.0e3, 3.0, 0.0, 1600.0, 0.0, 0], dtype=np.float32)
       
        self.act_min = np.array([0.0, -1.5708], dtype=np.float32) # min [alpha, sigma] 
        self.act_max = np.array([0.5236, 1.5708], dtype=np.float32) # max [alpha, sigma]
        self.obs_min = np.array([6380.0e3, -3.1416, -1.5708, 1000.0, -1.5708, -1.5708], dtype=np.float32) # min [r, theta, phi, v, gamma, psi]
        self.obs_max = np.array([6488.0e3, 3.1416, 1.5708, 6500.0, 1.5708, 1.5708], dtype=np.float32) #max [r, theta, phi, v, gamma, psi]

        self.action_space = spaces.Box(self.act_min, self.act_max, dtype=np.float32)
        self.observation_space = spaces.Box(self.obs_min, self.obs_max, dtype=np.float32)

        #self.render_mode = render_mode
        #self.renderer = Renderer(self.render_mode, self._render)

        #self.screen_width = 600
        #self.screen_height = 400
        #self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_done = None
        self.steps_beyond_terminated = None
        
    def gravity(self, r):
        """ Returns gravitational acceleration at a given planet radius based on quadratic model 
        
            For radius in meters, returns m/s**2
            For non-dimensional radius, returns non-dimensional gravity 
        """
        return (-self.planet.mu / (r ** 2))
    
    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        
        r, theta, phi, v, gamma, psi = self.state
        g0 = 9.8
        m = self.mass
        g = self.gravity(r)
        omega = self.planet.omega
        h = (r - self.planet.radius)
        rho, a = self.planet.atmosphere(h)
        M = v / a
        K = 5.188e-8        
        """
      
        if (action[0]-self.alpha_old) > 0.0017:
            alpha = self.alpha_old + 0.0017
        elif (action[0]-self.alpha_old) < -0.0017:
            alpha = self.alpha_old - 0.0017
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
        """
        alpha = action[0]
        sigma = action[1]
        
        [cl0, cl1, cl2, cl3] = [-0.2317, 0.0513, 0.2945, -0.1028]
        [cd0, cd1, cd2, cd3] = [0.024, 7.24e-4, 0.406, -0.323]
        
        cL = cl1 * (alpha * 180 / np.pi)   + cl2 * exp(cl3 * M) + cl0 # lift coefficient
        cD = cd1 * (alpha * 180 / np.pi) ** 2 + cd2 * exp(cd3 * M) + cd0 # drag coefficient
        
        #print(alpha, sigma, cL, cD)
        #aerodynamic heating
        qD = K * (rho ** 0.5) * (v ** 3)
        qD_max = 2e6
        #dynamic pressure
        pD = 0.5 * rho * v ** 2
        pD_max = 5e5
        
        L = pD * self.area * cL # lift force
        D = pD * self.area * cD # drag force
        #normal load
        nL = np.sqrt(L**2 + D**2) / (m * g0)
        nL_max = 10
        
        
        
        dr = v * sin(gamma)
        dtheta = (v * cos(gamma) * cos(psi)) / (r * cos(phi))
        dphi = (v * cos(gamma) * sin(psi)) / r
        dv = (-D )/ m - g * sin(gamma) + ((omega ** 2) * r * cos(phi)) * (sin(gamma) * cos(phi) - cos(gamma) * sin(phi) * sin(psi))
        dgamma = L * cos(sigma) / (m * v) + cos(gamma) * ((v ** 2 - g * r)/(r * v)) + 2 * omega * cos(phi) * cos(psi) + ((omega ** 2) * r * cos(phi)) * (cos(gamma) * cos(phi) + sin(gamma) * sin(phi) * sin(psi)) / v
        dpsi = L * sin(sigma) / (m * v * cos(gamma)) - (v / r) * cos(gamma) * cos(psi) * tan(phi) + 2 * omega * (tan(gamma) * cos(phi) * sin(psi) - sin(phi)) - (omega ** 2) * r * sin(phi) * cos(phi) * cos(psi) / (v * cos(gamma))

        if self.kinematics_integrator == "euler":
            r = r + self.tau * dr
            theta = theta + self.tau * dtheta
            phi = phi + self.tau * dphi
            v = v + self.tau * dv
            gamma = gamma + self.tau * dgamma
            psi = psi + self.tau * dpsi

        self.state = (r, theta, phi, v, gamma, psi)
        #print(h, cL, cD, v)
        terminated = bool(
            np.any(self.state < self.obs_min)
            or np.any(self.state > self.obs_max)
            or qD > qD_max
            or pD > pD_max 
            or nL > nL_max
        )

        if self.steps_beyond_terminated is None:
            reward = 0
            self.steps_beyond_terminated = 0
        elif self.steps_beyond_terminated == 0:
            logger.warn(
                "You are calling 'step()' even though this "
                "environment has already returned terminated = True. You "
                "should always call 'reset()' once you receive 'terminated = "
                "True' -- any further steps are undefined behavior."
            )
            self.steps_beyond_terminated += 1
            reward = 0.0
        else:
            
            if not terminated:
                reward = -5
                self.steps_beyond_terminated += 1
                if np.all(abs(self.state - self.final_state) <= np.array([0.1, 0.01, 0.01, 0.1, 0.01, 0.01])):
                    reward += 100000
                    terminated = True
                elif np.all(abs(self.state - self.final_state) <= np.array([5.0, 1.0, 1.0, 5.0, 1.0, 1.0])):
                    reward += 80000  
                elif np.all(abs(self.state - self.final_state) <= np.array([10.0, 1.5, 1.5, 100.0, 1.5, 1.5])):
                    reward += 50000
                elif np.all(abs(self.state - self.final_state) <= np.array([50.0, 2.0, 1.5, 150.0, 1.5, 1.5])):
                    reward += 45000
                elif np.all(abs(self.state - self.final_state) <= np.array([100.0, 2.0, 1.5, 200.0, 1.5, 1.5])):
                    reward += 40000
                elif np.all(abs(self.state - self.final_state) <= np.array([150.0, 2.0, 1.5, 300.0, 1.5, 1.5])):
                    reward += 35000 
                elif np.all(abs(self.state - self.final_state) <= np.array([200.0, 2.0, 1.5, 400.0, 1.5, 1.5])):
                    reward += 30000
                elif np.all(abs(self.state - self.final_state) <= np.array([250.0, 2.0, 1.5, 500.0, 1.5, 1.5])):
                    reward += 25000
                elif np.all(abs(self.state - self.final_state) <= np.array([300.0, 2.0, 1.5, 600.0, 1.5, 1.5])):
                    reward += 20000
                elif np.all(abs(self.state - self.final_state) <= np.array([350.0, 2.0, 1.5, 700.0, 1.5, 1.5])):
                    reward += 15000
                elif np.all(abs(self.state - self.final_state) <= np.array([400.0, 2.0, 1.5, 800.0, 1.5, 1.5])):
                    reward += 10000 
                elif np.all(abs(self.state - self.final_state) <= np.array([500.0, 2.0, 1.5, 850.0, 1.5, 1.5])):
                    reward += 8000
                elif np.all(abs(self.state - self.final_state) <= np.array([800.0, 2.5, 1.5, 900.0, 1.5, 1.5])):
                    reward += 7000
                elif np.all(abs(self.state - self.final_state) <= np.array([1000.0, 2.5, 1.5, 950.0, 1.5, 1.5])):
                    reward += 6000
                elif np.all(abs(self.state - self.final_state) <= np.array([2000.0, 2.5, 1.5, 1000.0, 1.5, 1.5])):
                    reward += 5000
                elif np.all(abs(self.state - self.final_state) <= np.array([2500.0, 2.5, 1.5, 1200.0, 1.5, 1.5])):
                    reward += 4500
                elif np.all(abs(self.state - self.final_state) <= np.array([3000.0, 2.5, 1.5, 2500.0, 1.5, 1.5])):
                    reward += 4000
                elif np.all(abs(self.state - self.final_state) <= np.array([3500.0, 2.5, 1.5, 2700.0, 1.5, 1.5])):
                    reward += 8500
                elif np.all(abs(self.state - self.final_state) <= np.array([4000.0, 2.5, 1.5, 3000.0, 1.5, 1.5])):
                    reward += 8500
                elif np.all(abs(self.state - self.final_state) <= np.array([4500.0, 2.5, 1.5, 3250.0, 1.5, 1.5])):
                    reward += 8000
                elif np.all(abs(self.state - self.final_state) <= np.array([5000.0, 2.5, 1.5, 3500.0, 1.5, 1.5])):
                    reward += 7000
                elif np.all(abs(self.state - self.final_state) <= np.array([6000.0, 2.5, 1.5, 4000.0, 1.5, 1.5])):
                    reward += 5000
                elif np.all(abs(self.state - self.final_state) <= np.array([08.0e3, 3.0, 2.0, 4000.0, 2.0, 2.0])):
                    reward += 2500
                elif np.all(abs(self.state - self.final_state) <= np.array([1.0e3, 3.0, 2.0, 4320.0, 2.0, 2.0])):
                    reward += 1000
                elif np.all(abs(self.state - self.final_state) <= np.array([2.0e3, 3.0, 2.0, 4350.0, 2.0, 2.0])):
                    reward += 900
                elif np.all(abs(self.state - self.final_state) <= np.array([3.0e3, 3.0, 2.0, 4375.0, 2.0, 2.0])):
                    reward += 800
                elif np.all(abs(self.state - self.final_state) <= np.array([4.0e3, 3.0, 2.0, 4420.0, 2.0, 2.0])):
                    reward += 700
                elif np.all(abs(self.state - self.final_state) <= np.array([5.0e3, 3.0, 2.0, 4450.0, 2.0, 2.0])):
                    reward += 600
                elif np.all(abs(self.state - self.final_state) <= np.array([6.0e3, 3.0, 2.0, 4480.0, 2.0, 2.0])):
                    reward += 500
                elif np.all(abs(self.state - self.final_state) <= np.array([7.0e3, 3.0, 2.0, 4510.0, 2.0, 2.0])):
                    reward += 400
                elif np.all(abs(self.state - self.final_state) <= np.array([8.0e3, 3.0, 2.0, 4550.0, 2.0, 2.0])):
                    reward += 300
                elif np.all(abs(self.state - self.final_state) <= np.array([9.0e3, 3.0, 2.0, 4580.0, 2.0, 2.0])):
                    reward += 290
                elif np.all(abs(self.state - self.final_state) <= np.array([10.0e3, 3.0, 2.0, 4600.0, 2.0, 2.0])):
                    reward += 280
                elif np.all(abs(self.state - self.final_state) <= np.array([10.0e3, 3.0, 2.0, 4610.0, 2.0, 2.0])):
                    reward += 270
                elif np.all(abs(self.state - self.final_state) <= np.array([12.5e3, 3.0, 2.0, 4630.0, 2.0, 2.0])):
                    reward += 250
                elif np.all(abs(self.state - self.final_state) <= np.array([13.0e3, 3.0, 2.0, 4650.0, 2.0, 2.0])):
                    reward += 230
                elif np.all(abs(self.state - self.final_state) <= np.array([13.5e3, 3.0, 2.0, 4670.0, 2.0, 2.0])):
                    reward += 220
                elif np.all(abs(self.state - self.final_state) <= np.array([14.0e3, 3.0, 2.0, 4700.0, 2.0, 2.0])):
                    reward += 200
                elif np.all(abs(self.state - self.final_state) <= np.array([14.5e3, 3.0, 2.0, 4710.0, 2.0, 2.0])):
                    reward += 190
                elif np.all(abs(self.state - self.final_state) <= np.array([15.0e3, 3.0, 2.0, 4725.0, 2.0, 2.0])):
                    reward += 180
                elif np.all(abs(self.state - self.final_state) <= np.array([20.0e3, 3.0, 2.0, 4740.0, 2.0, 2.0])):
                    reward += 170
                elif np.all(abs(self.state - self.final_state) <= np.array([20.0e3, 3.0, 2.0, 4750.0, 2.0, 2.0])):
                    reward += 160
                elif np.all(abs(self.state - self.final_state) <= np.array([20.5e3, 3.0, 2.0, 4760.0, 2.0, 2.0])):
                    reward += 150
                elif np.all(abs(self.state - self.final_state) <= np.array([22.0e3, 3.0, 2.0, 4775.0, 2.0, 2.0])):
                    reward += 140
                elif np.all(abs(self.state - self.final_state) <= np.array([23.0e3, 3.0, 2.0, 4800.0, 2.0, 2.0])):
                    reward += 120
                elif np.all(abs(self.state - self.final_state) <= np.array([24.0e3, 3.0, 2.0, 4810.0, 2.0, 2.0])):
                    reward += 110
                elif np.all(abs(self.state - self.final_state) <= np.array([25.0e3, 3.0, 2.0, 4820.0, 2.0, 2.0])):
                    reward += 100
                elif np.all(abs(self.state - self.final_state) <= np.array([26.0e3, 3.0, 2.0, 4825.0, 2.0, 2.0])):
                    reward += 90
                elif np.all(abs(self.state - self.final_state) <= np.array([30.0e3, 3.0, 2.0, 4840.0, 2.0, 2.0])):
                    reward += 80
                elif np.all(abs(self.state - self.final_state) <= np.array([30.0e3, 3.0, 2.0, 4855.0, 2.0, 2.0])):
                    reward += 70
                elif np.all(abs(self.state - self.final_state) <= np.array([35.0e3, 3.0, 2.0, 4865.0, 2.0, 2.0])):
                    reward += 60
                elif np.all(abs(self.state - self.final_state) <= np.array([38.5e3, 3.0, 2.0, 4865.0, 2.0, 2.0])):
                    reward += 50
                elif np.all(abs(self.state - self.final_state) <= np.array([40.0e3, 3.0, 2.0, 4870.0, 2.0, 2.0])):
                    reward += 40
                elif np.all(abs(self.state - self.final_state) <= np.array([40.0e3, 3.0, 2.0, 4885.0, 2.0, 2.0])):
                    reward += 30
                elif np.all(abs(self.state - self.final_state) <= np.array([46.85e3, 3.0, 2.0, 4890.0, 2.0, 2.0])):
                    reward += 20
                elif np.all(abs(self.state - self.final_state) <= np.array([48.0e3, 3.0, 2.0, 4895.0, 2.0, 2.0])):
                    reward += 15
                elif np.all(abs(self.state - self.final_state) <= np.array([50.0e3, 3.0, 2.0, 4900.0, 2.0, 2.0])):
                    reward += 1
                elif np.any(abs(self.state - self.final_state) >= np.array([50.0e4, 6.3, 3.5, 4900.0, 3.5, 3.5])):
                    reward += -10
                elif np.any(abs(self.state - self.final_state) >= np.array([55.0e4, 6.3, 3.5, 5000.0, 3.5, 3.5])):
                    reward += -500
                elif np.any(abs(self.state - self.final_state) >= np.array([60.0e4, 6.3, 3.5, 5500.0, 3.5, 3.5])):
                    reward += -1000
            else:  
                reward = -200000
        #print(abs(self.state - self.final_state), reward)
        #self.renderer.render_step()
        print(reward, h, v, dv, L, D)
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
class ActionConstraint(gym.ActionWrapper):
    def __init__(self, env, updated_act):
        super().__init__(env)
        self.updated_act = updated_act
        self.action_space = self.
        self.act_min = np.array([-30, -90], dtype=np.float32) # min [alpha, sigma] 
        self.act_max = np.array([30, 90], dtype=np.float32) # max [alpha, sigma]

        def action(self):            
            alpha_old = self.action_space[0]
            sigma_old = self.action_space[1]            
            self.action = spaces.Box(self.act_min, self.act_max, dtype=np.float32)
            alpha = self.action[0]
            sigma = self.action[1]
            if (abs(alpha-alpha_old) > 5):
                alpha = alpha_old
            elif (abs(sigma-sigma_old) > 30):
                sigma = sigma_old
            elif (abs(alpha-alpha_old) < 5 and abs(sigma-sigma_old) < 30):
                alpha = alpha
                sigma = sigma
            
            self.updated_act = [alpha, sigma]
            return self.updated_act
        
    
    def aerodynamic_coefficients(self, M, alpha):
        r, theta, phi, v, gamma, psi = self.state
        h = (r - self.planet.radius) / self.dist_scale
        rho, a = self.planet.atmosphere(h * self.dist_scale)
        self.M = v * self.vel_scale / a
        self.alpha = self.step().action[0]
        # Returns aero coefficients CD and CL. Supports ndarray Mach numbers.
        [cl0, cl1, cl2, cl3] = [-0.2317, 0.0513, 0.2945, -0.1028]
        [cd0, cd1, cd2, cd3] = [0.024, 7.24e-4, 0.406, -0.323]
        
        cL = cl1 * self.alpha + cl2 * exp(cl3 * self.M) + cl0
        cD = cd1 * self.alpha ** 2 + cd2 * exp(cd3 * self.M) + cd0
        LoD = cL / cD
        return cD, cL
    
    def ballistic_coefficients(self, mass=907.2, area=0.4839):
        self.mass = mass
        self.cD = self.aerodynamic_coefficients(M, alpha)[0]
        self.area = area
        
        bc = mass/(self.cD * area)
        return bc
    
        
    
    def action_old(self, action):
        assert self.action_space.contains(action)
        
        alpha_old = action[0]
        sigma_old = action[1]
        if abs(alpha - alpha_old) > 0.3490:
            alpha = alpha_old
        if abs(sigma - sigma_old) > 0.5235:
            sigma = sigma_old
        
        return alpha_old, sigma_old
        
       
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




