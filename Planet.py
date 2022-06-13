#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

