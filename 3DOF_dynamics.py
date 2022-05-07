import numpy as np
from numpy import sin, cos, tan

def __entry_3dof(self, x, t, u, D, L, omega):
    r, theta, phi, v, gamma, psi =  x
    sigma, throttle, mu = u
    g = self.gravity(r)
    
    dr = v*sin(gamma)
    dtheta = v*cos(gamma)*cos(psi)/r*cos(phi)
    dphi = v*cos(gamma)*sin(psi)/r
    dv = -D/m - g*sin(gamma) + omega**2*r*cos(phi)*(sin(gamma)*cos(phi) - cos(gamma)*sin(phi)*sin(psi))
    dgamma = L*cos(sigma)/m*v + cos(gamma)*((v**2-g*r)/r*v) + 2*omega*sin(psi)*cos(phi) + omega**2*r*cos(phi)*(cos(phi)*cos(gamma)+sin(phi)*sin(psi)*sin(gamma)) 
    dpsi = L*sin(sigma)/m*v*cos(gamma) - v*cos(gamma)*cos(psi)*tan(phi)/r + 2*omega*(tan(gamma)*cos(phi)*sin(psi)-sin(phi)) - omega**2*r*sin(phi)*cos(phi)*cos(psi)/cos(gamma)
    return np.array([dr, dtheta, dphi, dv, dgamma, dpsi])
