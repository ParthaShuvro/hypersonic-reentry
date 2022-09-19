#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import sin, cos, tan
from Entry_Vehicle import EntryVehicle
import Planet

class Entry(object):
    """  Basic equations of motion for unpowered and powered flight through an atmosphere. """

    def __init__(self, PlanetModel=Planet('Earth'), VehicleModel=EntryVehicle(), Coriolis=False, Powered=False, Energy=False, Altitude=False, DifferentialAlgebra=False, Scale=False, Longitudinal=False, Velocity=False):


        
        self.planet = PlanetModel
        self.vehicle = VehicleModel
        self.powered = Powered
        self.drag_ratio = 1
        self.lift_ratio = 1
        self.nx = 6  # [r,lon,lat,v,gamma,psi]
        self.nu = 2  # bank command, angle of attack
        self.__jacobian = None  # If the jacobian method is called, the Jacobian object is stored to prevent recreating it each time. It is not constructed by default.
        self.__jacobianb = None
        self._da = DifferentialAlgebra
        self.planet._da = DifferentialAlgebra

        #todo: Non-dimensionalizing the states
        if Scale:
            self.dist_scale = self.planet.radius
            self.acc_scale = self.planet.mu/(self.dist_scale**2)
            self.time_scale = np.sqrt(self.dist_scale/self.acc_scale)
            self.vel_scale = np.sqrt(self.dist_scale*self.acc_scale)
            self.long_scale = self.theta_us/pi
            self.lat_scale = self.phi_us/pi
            self.fpa_scale = self.gamma_us/pi
            self.ha_scale = self.psi_us/pi
            self.mass_scale = 1

        else:  # No scaling
            self.dist_scale = 1
            self.acc_scale = 1
            self.time_scale = 1
            self.vel_scale = 1
        

        self.dyn_model = self.__entry_3dof

    def update_ratios(self, LR, DR):
        self.drag_ratio = DR
        self.lift_ratio = LR

    def DA(self, bool=None):
        if bool is None:
            return self._da
        else:
            self._da = bool
            self.planet._da = bool

# Dynamics model
        
    def __entry_3dof(self, s, alpha, sigma, t):
        r, theta, phi, v, gamma, psi =  self.s
        g = self.gravity(r)
        omega = self.planet.omega
        h = (r - self.planet.radius)/self.dist_scale
        rho, a = self.planet.atmosphere(h*self.dist_scale)
        M = v*self.vel_scale/s_s
        alpha = self.Control.alpha
        sigma = self.Control.sigma
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

        return np.array([dr, dtheta, dphi, dv, dgamma, dpsi])
    
    def aeroforces(self, r, v, m):
        """  Returns the aerodynamic forces acting on the vehicle at a given radius, velocity and mass. """

        h = r - self.planet.radius
        rho, a = self.planet.atmosphere(h)
        M = v/a
        # cD, cL = self.vehicle.aerodynamic_coefficients(M)
        cD, cL = self.vehicle.aerodynamic_coefficients(v)
        f = 0.5*rho*self.vehicle.area*v**2/m
        L = f*cL*self.lift_ratio
        D = f*cD*self.drag_ratio
        return L, D

    def gravity(self, r):
        """ Returns gravitational acceleration at a given planet radius based on quadratic model 
        
            For radius in meters, returns m/s**2
            For non-dimensional radius, returns non-dimensional gravity 
        """
        return self.planet.mu/(r*self.dist_scale)**2/self.acc_scale

