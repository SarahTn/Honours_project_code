# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 16:31:52 2022

@author: ST
"""
# ADC
import numpy as np

def linear1(x, m):
    y = -m*x
    return y

def linear2(x, m, b):
    y = -m*x + b
    return y

# IVIM
def ivim(x, D, Dstar, f, S0):
    S = S0*((1-f)*np.exp(-x*D) + f*np.exp(-x*Dstar))    
    return S

def ivim_2step(x, D, Dstar, A, B):
    S = A*np.exp(-x*D) + B*np.exp(-x*(Dstar))
    return S

def ivim_simp(x, D, f, S0):
    S = S0*(1 - f)*np.exp(-x*D)
    return S


# Kurtosis 
def kurtosis(x, Dapp, Kapp, S0):
    S = S0*np.exp(-x*Dapp + (1/6)*(x**2)*(Dapp**2)*Kapp)
    # S = np.exp(-x*Dapp + (1/6)*(x**2)*(Dapp**2)*Kapp)
    return S

def lin_kurtosis(x, Dapp, Kapp, lnS0):
    # S = S0*np.exp(-x*Dapp + (1/6)*(x**2)*(Dapp**2)*Kapp)
    lnS = lnS0 -x*Dapp + (1/6)*((x**2)*(Dapp**2)*Kapp)
    return lnS








