#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import Control
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

    def __init__(self, area=0.4839, mass=907.2):
        # need to put real value for cav thrust, ISP
        self.area = area
        self.mass = mass
        self.g0 = 9.81

    def aerodynamic_coefficients(self, M, alpha):
        self.M = self.Control.mach()
        self.alpha = self.Control.alpha()
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

