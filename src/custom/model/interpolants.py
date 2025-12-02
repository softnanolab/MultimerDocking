"""
Module contains functionalities for stochastic interpolants.
"""


import torch


class Interpolant:
    def __init__(self):
        return None
    
    def gamma(self, t):
        """Noise coefficient along the path, and derivative"""
        return None, None
    
    def interpolant(self, t, x_0, x_1):
        """Computes interpolant I(t, x_0, x_1), and derivative"""
        return None, None

    def compute_x_t(self, t, x_0, x_1, z):
        """Computes x_t = I(t, x_0, x_1) + gamma(t)*z"""
        gamma_t, _ = self.gamma(t)
        I, _ = self.interpolant(t, x_0, x_1)
        x_t = I + gamma_t*z
        return x_t

    def velocity_target(self, t, x_0, x_1, z):
        """Computes velocity target = dx_t/dt"""
        _, d_interpolant = self.interpolant(t, x_0, x_1)
        _, d_gamma = self.gamma(t)
        v_t = d_interpolant + d_gamma*z
        return v_t


class LinearInterpolant(Interpolant):
    def __init__(self):
        return None
    
    def gamma(self, t):
        """Noise coefficient along the path, and derivative"""
        gamma_t = torch.sqrt(2*t*(1-t))
        d_gamma_t = (1-2*t)/torch.sqrt(2*t*(1-t))
        return gamma_t, d_gamma_t
    
    def alpha(self, t):
        """Coefficient of x_0, and derivative"""
        return 1 - t, -1

    def beta(self, t):
        """Coefficient of x_1, and derivative"""
        return t, 1

    def interpolant(self, t, x_0, x_1):
        """Computes interpolant I(t, x_0, x_1), and derivative"""
        alpha, d_alpha = self.alpha(t)
        beta, d_beta = self.beta(t)
        I = alpha*x_0 + beta*x_1
        d_I = d_alpha*x_0 + d_beta*x_1
        return I, d_I