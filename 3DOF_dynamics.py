#!/usr/bin/env python
# coding: utf-8

# In[17]:


class EntryVehicle(object):
    
    '''
    Defines an EntryVehicle class:
    members:
        area - the effective area, m^2
        CD   - a multiplicative offset for the vehicle's drag coefficient
        CL   - a multiplicative offset for the vehicle's lift coefficient
        SRP-related parameters:
        Thrust - the total thrust of the vehicle BEFORE efficiency losses, cant angle considerations, etc, Newtons
        ThrustFactor - mean(cos(cantAngles)*(1-superSonicEfficiencyLosses)), non-dimensional
        Isp - specific impulse, s
        g0 - sea level Earth gravity, used in mass rate of change, m/s^2
    methods:
        mdot(throttle) - computes the mass rate of change based on the current throttle setting
        aerodynamic_coefficients(Mach, alpha) - computes the values of CD and CL for the current Mach values, angle of attack
        BC(mass, Mach) - computes the vehicle's ballistic coefficient as a function of its mass. Drag coefficient is calculated by default at Mach 24.
    '''

    def __init__(self, area=0.4839, mass=907.2, Thrust=60375., Isp=260., ThrustFactor=1.):
        self.area = area
        self.mass = mass
        self.Thrust = Thrust
        self.ThrustFactor = ThrustFactor
        self.ThrustApplied = self.Thrust*self.ThrustFactor
        self.g0 = 9.81
        self.isp = Isp
        self.ve = self.g0*self.isp
        cD,cL = self.aerodynamic_coefficients(20, 10)
        self.LoD = cL/cD

    def mdot(self, throttle):
        """ Returns the mass flow rate for a given throttle setting. """
        return -self.Thrust*throttle/(self.ve)

    def aerodynamic_coefficients(self, M, alpha):
        from math import exp
        """ Returns aero coefficients CD and CL. Supports ndarray Mach numbers. """
        [cl0, cl1, cl2, cl3] = [-0.2317, 0.0513, 0.2945, -0.1028]
        [cd0, cd1, cd2, cd3] = [0.024, 7.24*exp(-4), 0.406, -0.323]
        
        cL = cl1*alpha + cl2*exp(cl3*M) + cl0
        cD = cd1*alpha**2 + cd2*exp(cd3*M) + cd0

        return cD, cL
    
    def ballistic_coefficients(self, mass, cD, area):
        self.mass = mass
        self.cD = self.aerodynamic_coefficients(M, alpha)[0]
        self.area = area
        
        bc = mass/(cD*area)
        return bc


# In[18]:


import isacalc as isa
atmos = isa.get_atmosphere

class Planet:
    def __init__(self, name='Earth', rho0=0, scaleHeight=0, model='exp', da=False):

        self.name = name.capitalize()
        self._da = da  # Differential algebraic inputs

        if self.name == 'Earth':
            self.radius = 6378.1e3
            self.omega = 7.292115e-5
            self.mu = 3.98600e14
            
            self.atmosphere= self.__atmos_earth

        elif self.name == 'Mars':
            self.radius = 3396.2e3
            self.omega = 7.095e-5
            self.mu = 4.2830e13

            if model is 'exp':
                self.rho0 = (1+rho0)*0.0158
                self.scaleHeight = (1+scaleHeight)*9354.5
                self.atmosphere = self.__exp_model_mars

            else:
                # Sample MG and store interpolators for density and speed of sound
                self.atmosphere = self.__MG_model_mars
        else:
            print('Input planet name, '+ self.name +', is not valid')
    def __atmos_earth(self, h):
        atmosphere = isa.get_atmosphere
        T, P, rho, a, mu = isa.calculate_at_h(h, atmosphere)
        return rho, a
        
    def __exp_model_mars(self, h):
        ''' Defines an exponential model of the atmospheric density and local speed of sound as a function of altitude. '''
        if self._da:
            from pyaudi import exp
            scalar=False
            try:
                h[0]
            except:
                scalar=True
                h = [h]
            #Density computation:
            rho = [self.rho0*exp(-hi/self.scaleHeight) for hi in h]

            # Local speed of sound computation:
            coeff = [223.8, -0.2004e-3, -1.588e-8, 1.404e-13]
            a = [sum([c*hi**i for i,c in enumerate(coeff)]) for hi in h]
            if scalar:
                a = a[0]
                rho = rho[0]

        else:
            from numpy import exp
            #Density computation:
            rho = self.rho0*exp(-h/self.scaleHeight)

            # Local speed of sound computation:
            coeff = [223.8, -0.2004e-3, -1.588e-8, 1.404e-13]
            a = sum([c*h**i for i,c in enumerate(coeff)])

        return rho,a

    def __MG_model_mars(self, h):
        ''' Interpolates data from an MG profile '''
        return self.density(h),self.speed_of_sound(h)

    def updateMG(date=[10,29,2018], latitude=0, longitude=0, dustTau=0, rpscale=0):
        ''' Calls MG '''
        return


    def heading(self, lon1, lat1, lon2, lat2):
        # Given two points on a spherical planet, compute the heading that links that along a great circle arc
        # Reference 2.39 - 2.44 in Joel's dissertation 
        from numpy import pi, sin, cos, arccos, sign, abs, arcsin

        d1n = pi/2 - lat1
        d2n = pi/2 - lat2
        # From diss
        cd12 = sin(d1n)*sin(d2n)*cos(lon2-lon1) + cos(d1n)*cos(d2n)
        d12 = arccos(cd12)

        # From paper
        # d12 = 2*arcsin(sin(0.5*(lat1-lat2))**2 + cos(lat1)*cos(lat2)*sin(0.5*(lon1-lon2))**2)
        # cd12 = cos(d12)
        
        cphi = (sin(lat2)-sin(lat1)*cd12) / (cos(lat1)*sin(d12))
        phi = sign(lon2-lon1) * arccos(cphi)
        # if abs(lon1-lon2) < 1e-4:
        #     if lat2 > lat1:
        #         phi = 0
        #     else:
        #         phi = pi 
        psi12 = pi/2 - phi 
        return psi12


    def range(self, lon0, lat0, heading0, lonc, latc, km=False):
        '''Computes the downrange and crossrange between two lat/lon pairs with a given initial heading.'''
        # from numpy import arccos, arcsin, sin, cos, pi, nan_to_num, zeros_like
        from numpy import pi, nan_to_num, zeros_like, real
        import numpy as np
        from pyaudi import gdual_double as gd

        dual = False 
        for thing in [lon0, lat0, heading0, lonc, latc]:
            if isinstance(thing, gd):
                dual = True
                break 
        if dual:
            from pyaudi import sin, cos
            from pyaudi import asin as arcsin
            from pyaudi import acos as arccos
            from Utils.DA import sign

        else:
            from numpy import sin, cos, arcsin, arccos, sign

        d13 = arccos(sin(latc)*sin(lat0)+cos(latc)*cos(lat0)*cos(lonc-lon0))
        # if not isinstance(d13, gd) and np.abs(d13) < 1e-4:
        #     return 0,0
        psi12 = heading0
        PHI = sign(lonc-lon0)*arccos( (sin(latc) - sin(lat0)*cos(d13))/(cos(lat0)*sin(d13)) )
        psi13 = pi/2 - PHI  # This is the desired heading angle 

        CR = arcsin(sin(d13)*sin(psi12-psi13))
        DR = self.radius*arccos(cos(d13)/cos(CR))
        CR *= self.radius
        try:
            DR[np.isnan(PHI)] = 0
            CR[np.isnan(PHI)] = 0
        except TypeError:
            pass 

        if km:
            return DR/1000., CR/1000.
        else:
            return DR, CR

    def coord(self, lon0, lat0, heading0, dr, cr):
        '''Computes the coords of a target a given downrange and crossrange from an initial location and heading.'''
        from numpy import arccos, arcsin, sin, cos, pi
        # from pyaudi import sin, cos
        # from pyaudi import asin as arcsin
        # from pyaudi import acos as arccos

        LF = arccos(cos(dr/self.radius)*cos(cr/self.radius))
        zeta = arcsin(sin(cr/self.radius)/sin(LF))
        lat = arcsin(cos(zeta-heading0+pi/2.)*cos(lat0)*sin(LF)+sin(lat0)*cos(LF))
        lon = lon0 + arcsin(sin(zeta-heading0+pi/2)*sin(LF)/cos(lat))
        return lon, lat


def getDifference(rho0, scaleHeight):
    import numpy as np

    nominal = Planet()
    dispersed = Planet(rho0=rho0, scaleHeight=scaleHeight)

    h = np.linspace(0, 127e3, 1000) # meters
    rho_nom = [nominal.atmosphere(H)[0] for H in h]
    rho_dis = [dispersed.atmosphere(H)[0] for H in h]
    diff = np.array(rho_dis)-np.array(rho_nom)
    perdiff = 100*diff/np.array(rho_nom)
    return perdiff


def compare():
    from itertools import product
    import matplotlib.pyplot as plt
    import numpy as np

    n = 2
    rho0 = np.linspace(-0.20, 0.20, n)
    sh = np.linspace(-0.025, 0.01, n)
    h = np.linspace(0, 127, 1000) # kmeters
    
    plt.figure()
    for rho, s in product(rho0, sh):
        perDiff = getDifference(rho, s)
        plt.plot(h, perDiff, label="\rho={}, \hs={}".format(rho, s))
    plt.legend(loc='best')
    plt.xlabel('Altitude (km)')
    plt.ylabel('Density variation (%)')
    plt.show()


if __name__ == "__main__":
    import numpy as np 
    import matplotlib.pyplot as plt 
    # compare()
    print(np.degrees(Planet().coord(*np.radians([66.026, 21.481, 90-103.6]), -48e3, 0)))
    # print(Planet().coord(*np.radians([66.026, 21.481, 90-103.6]), -48e3, 0))

    lat = np.linspace(-88, 88, 100)
    #for phi in lat:
        #plt.plot(lat, Planet().heading(0, 0, 600/3397, np.radians(lat))*180/3.14)
        #plt.show()
    # for phi in lat:
        # print(Planet().heading(0, 0, phi*3.14/180, phi*3.14/180)*180/3.14)


# In[20]:


import numpy as np
from numpy import sin, cos, tan
#import EntryVehicle
#import Planet

class Entry(object):
    """  Basic equations of motion for unpowered and powered flight through an atmosphere. """

    def __init__(self, PlanetModel=Planet('Earth'), VehicleModel=EntryVehicle(), Coriolis=False, Powered=False, Energy=False, Altitude=False, DifferentialAlgebra=False, Scale=False, Longitudinal=False, Velocity=False):


        
        self.planet = PlanetModel
        self.vehicle = VehicleModel
        self.powered = Powered
        self.drag_ratio = 1
        self.lift_ratio = 1
        self.nx = 6  # [r,lon,lat,v,gamma,psi]
        self.nu = 6  # bank command, throttle, thrust angle, roll, pitch, yaw rates
        self.__jacobian = None  # If the jacobian method is called, the Jacobian object is stored to prevent recreating it each time. It is not constructed by default.
        self.__jacobianb = None
        self._da = DifferentialAlgebra
        self.planet._da = DifferentialAlgebra

        # Non-dimensionalizing the states
        if Scale:
            self.dist_scale = self.planet.radius
            self.acc_scale = self.planet.mu/(self.dist_scale**2)
            self.time_scale = np.sqrt(self.dist_scale/self.acc_scale)
            self.vel_scale = np.sqrt(self.dist_scale*self.acc_scale)
            self.mass_scale = 1
            self._scale = np.array([self.dist_scale, 1, 1, self.vel_scale, 1, 1, 1])

        else:  # No scaling
            self.dist_scale = 1
            self.acc_scale = 1
            self.time_scale = 1
            self.vel_scale = 1
            self._scale = np.array([self.dist_scale, 1, 1, self.vel_scale, 1, 1, 1])

        self.dyn_model = self.__entry_3dof
        self.use_energy = Energy
        if self.use_energy:
            self.dE = None
        else:
            self.dE = 1

        self.use_altitude = Altitude
        if self.use_altitude:
            self.dE = None

    def update_ratios(self, LR, DR):
        self.drag_ratio = DR
        self.lift_ratio = LR

    def DA(self, bool=None):
        if bool is None:
            return self._da
        else:
            self._da = bool
            self.planet._da = bool

    def ignite(self):
        """ Ignites the engines to begin powered flight. """
        self.powered = True

    def dynamics(self, u):
        if self.powered:
            return lambda x,t: self.dyn_model(x, t, u)+self.__thrust_3dof(x, u)

        else:
            return lambda x,t: self.dyn_model(x, t, u)

# Dynamics model
        
    def __entry_3dof(self, x, t):
        r, theta, phi, v, gamma, psi =  x
        g = self.gravity(r)
        h = r - self.planet.radius/self.dist_scale
        rho, a = self.planet.atmosphere(h*self.dist_scale)
        M = v*self.vel_scale/a
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
    
    def __thrust_3dof(self, x, u):
        if self._da:
            from pyaudi import sin, cos, tan
        else:
            from numpy import sin, cos, tan
        r,theta,phi,v,gamma,psi,m = x
        sigma,throttle,thrustAngle = u

        return np.array([0,0,0,self.vehicle.ThrustApplied*throttle*cos(sigma)*cos(thrustAngle-gamma)/m, self.vehicle.ThrustApplied*throttle*sin(thrustAngle-gamma)/(m*v), self.vehicle.ThrustApplied*throttle*cos(thrustAngle-gamma)*sin(sigma)/(cos(gamma)*m*v**2), self.vehicle.mdot(throttle)])/self.dE

    def _bank(self, x):
        """ Internal function used for jacobian of bank rate """

        r, theta, phi, v, gamma, psi, m, sigma, T, mu, sigma_dot = x

        if self.use_energy:
            h = r - self.planet.radius/self.dist_scale
            g = self.gravity(r)
            rho,a = self.planet.atmosphere(h*self.dist_scale)
            M = v*self.vel_scale/a
            cD,cL = self.vehicle.aerodynamic_coefficients(M)
            f = np.squeeze(0.5*rho*self.vehicle.area*(v*self.vel_scale)**2/m)/self.acc_scale
            D = f*cD*self.drag_ratio
            return sigma_dot/(-v*D)
        else:
            return sigma_dot

    def bank_jacobian(self, x, u, sigma_dot):
        # Superior DA approach - much faster
        from Utils import DA as da
        vars = ['r','theta','phi','v','fpa','psi','m','bank','T','mu','bank_rate']
        X = np.concatenate((x,u))
        X = np.append(X, sigma_dot)
        X = da.make(X, vars, 1, array=True)
        F = self._bank(X)
        return da.jacobian([F], vars)


    # Utilities
    def altitude(self, r, km=False):
        """ Computes the altitude from radius """
        if km:
            return (r-self.planet.radius)/1000.
        else:
            return r-self.planet.radius

    def radius(self, h):
            return h + self.planet.radius

    def energy(self, r, v, Normalized=True):
        """ Computes the current energy at a given radius and velocity. """

        E = 0.5*v**2 + self.planet.mu/self.planet.radius-self.planet.mu/r
        if Normalized:
            return (E-E[0])/(E[-1]-E[0])
        else:
            return E

    def scale(self, state):
        """Takes a state or array of states in physical units and returns the non-dimensional verison """
        shape = np.asarray(state).shape
        if len(shape)==1 and shape[0]==self.nx:
            return state/self._scale
        else:
            return state/np.tile(self._scale, (shape[0],1))

    def scale_time(self, time):
        return time/self.time_scale

    def unscale(self, state):
        """ Converts unitless states to states with units """
        shape = np.asarray(state).shape
        if len(shape) == 1 and shape[0] == self.nx:
            return state*self._scale
        else:
            return state*np.tile(self._scale, (shape[0], 1))

    def unscale_time(self, time):
        return time*self.time_scale

    def jacobian(self, x, u, hessian=False, vectorized=True):
        """ Returns the full jacobian of the entry dynamics model. 
            The dimension will be [nx, nx+nu].
        """
        return self._jacobian_pyaudi(x, u, hessian, vectorized)

    def _jacobian_ndt(self, x, u):
        ''' Jacobian computed via numdifftools '''
        if self.__jacobian is None:
            from numdifftools import Jacobian
            self.__jacobian = Jacobian(self.__dynamics(), method='complex')

        state = np.concatenate((x, u))
        if self.use_velocity:
            state = np.concatenate((x[:-1], u, x[-1, None]))

        return self.__jacobian(state)

    def _jacobian_pyaudi(self, x, u, hessian=False, vectorized=False):
        ''' Jacobian computed via pyaudi '''

        da_setting = self.DA()
        self.DA(True)

        from Utils import DA as da
        vars = ['r','theta','phi','v','fpa','psi','m','bank','T','mu']
        if vectorized:
            xu = np.concatenate((x.T, u.T))
        else:
            xu = np.concatenate((x, u))

        X = da.make(xu, vars, 1+hessian, array=True, vectorized=vectorized)
        f = self.__dynamics()(X)
        if hessian:
            J = da.jacobian(f, vars)
            H = da.vhessian(f, vars)
            self.DA(da_setting)
            return J, H
        else:
            J = da.jacobian(f, vars)
            self.DA(da_setting)
            return J

    def __dynamics(self):
        ''' Used in jacobian. Returns an object callable with a single combined state '''

        if self.powered:
            return lambda xu: self.dyn_model(xu[0:self.nx], xu[-1], xu[self.nx:self.nx+self.nu])+self.__thrust_3dof(xu[0:self.nx], xu[self.nx:self.nx+self.nu])
        else:
            return lambda xu: self.dyn_model(xu[0:self.nx], xu[-1], xu[self.nx:self.nx+self.nu])

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


# In[ ]:




