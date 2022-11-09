#!/usr/bin/env python
# coding: utf-8

# In[8]:


from Planet import Planet
import HRVEnv
import numpy as np
import math
class EntryVehicle(object):
    
    '''
    Defines an EntryVehicle class:
    members:
        area - the effective area, m^2
        CD   - a multiplicative offset for the vehicle's drag coefficient
        CL   - a multiplicative offset for the vehicle's lift coefficient
        g0 - sea level Earth gravity, used in mass rate of change, m/s^2
    methods:
        aerodynamic_coefficients(Mach, alpha) - computes the values of CD and CL for the current Mach values, angle of attack
        BC(mass, Mach) - computes the vehicle's ballistic coefficient as a function of its mass. Drag coefficient is calculated by default at Mach 24.
    '''

    def __init__(self, area=0.4839, mass=907.2, PlanetModel=Planet('Earth')):
        # need to put real value for cav thrust, ISP
        self.planet = PlanetModel
        self.area = area
        self.mass = mass
        self.g0 = 9.81
        self.dist_scale = self.planet.radius
        self.acc_scale = self.planet.mu/(self.dist_scale**2)
        self.time_scale = np.sqrt(self.dist_scale/self.acc_scale)
        self.vel_scale = np.sqrt(self.dist_scale*self.acc_scale)
        self.mass_scale = 1

    def aerodynamic_coefficients(self, M, alpha):
        r, theta, phi, v, gamma, psi = self.HRVEnv.state
        h = (r - self.planet.radius)/self.dist_scale
        rho, a = self.planet.atmosphere(h*self.dist_scale)
        self.M = self.v*self.vel_scale/a
        self.alpha = self.HRVEnv.action[0] # not sure if this works
        from math import exp
        """ Returns aero coefficients CD and CL. Supports ndarray Mach numbers. """
        [cl0, cl1, cl2, cl3] = [-0.2317, 0.0513, 0.2945, -0.1028]
        [cd0, cd1, cd2, cd3] = [0.024, 7.24*exp(-4), 0.406, -0.323]
        
        cL = cl1*self.alpha + cl2*exp(cl3*self.M) + cl0
        cD = cd1*self.alpha**2 + cd2*exp(cd3*self.M) + cd0
        LoD = cL/cD
        return cD, cL
    
    def ballistic_coefficients(self, mass=907.2, area=0.4839):
        self.mass = mass
        self.cD = self.aerodynamic_coefficients(M, alpha)[0]
        self.area = area
        
        bc = mass/(self.cD*area)
        return bc


# In[ ]:





# In[ ]:




