#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 14:55:45 2022

@author: a98alvin
"""


def CC_eqn(T):
    import numpy as np
    es0 = 6.11 #hPa
    Lv = 2.501 * 10 **6 #J/kg
    Rv = 461 # J/kg/K
    T0 = 273 # K
    
    es = es0 * np.exp((Lv/Rv) * ((1/T0)-(1/T)))
    return es

def potential_temp(T,P):
    P0 = 1000 # hPa
    R = 287 # J/kgK
    cp = 1004 # J/kgK
    
    theta = T * (P0/P)**(R/cp)
    return theta

def Temp_LCL_Ice(P0,T0,w): # kPa, K, kg/kg
    import numpy as np
    A = 3.41 * (10**9) # kPa
    B = 6.13 * (10**3) # K
    E = 287/461
    k = 287/1004
    Tc1 = 250 # K
    
    denom = np.log(((A*E)/(w*P0))*((T0/Tc1)**(1/k)))
    Tc2 = B/denom
    Tc = Tc2
    for i in range(0,10):
        Tc = B/(np.log(((A*E)/(w*P0))*((T0/Tc)**(1/k))))
    return Tc

def Temp_LCL_Vapor(P0,T0,w): # kPa, K, kg/kg
    import numpy as np    
    A = 2.53 * (10**8) # kPa
    B = 5.42 * (10**3) # K
    E = 287/461
    k = 287/1004    
    Tc1 = 250 # K

    denom = np.log(((A*E)/(w*P0))*((T0/Tc1)**(1/k)))
    Tc2 = B/denom
    Tc = Tc2
    for i in range(0,100):
        Tc = B/(np.log(((A*E)/(w*P0))*((T0/Tc)**(1/k))))
    return Tc

def air_density(Ts,Td,Ps): # C, C, hPa
    R = 287 # J/kgK
    Rv = 461 # J/kgK
    TsK = Ts + 273
    TdK = Td + 273
    PsPa = Ps * 100
    e = CC_eqn(TdK) * 100
    rho = ((PsPa-e)/(R * TsK)) + (e/(Rv * TsK))
    return rho

def mixing_ratio_from_e_approx(e, P):
    Rd = 287
    Rv = 461
    w = ((Rd/Rv) * e)/(P-e)
    return w

def virtual_temp(T,w):
    E = 287/461
    Tv = (T*(1+w/E))/(1 + w)
    return Tv

def equivalent_temp(T, w): # K, kg/kg
    Lv = 2.501 * 10 ** 6 # J/kg/K
    cpd = 1004 # J/kg/K
    
    Te = T + ((Lv/cpd) * w)
    return Te

def wet_bulb_temp(T, f, w): # K, fraction, kg/kg
    cp = 1004 # J/kg/K
    Lv = 2.501 * 10**6 # J/kg
    Rv = 461 # J/kgK
    
    Bottom = (cp/Lv) + ((Lv*w)/(f*Rv*(T**2)))
    Top = (((1-f)/f) * w)

    Tw = T - Top/Bottom
    return Tw

def find_closest(arr, val):
    idx = np.abs(arr - val).argmin()
    return arr[idx]
def lift_mountain_down_vapor(w,P0,T0, Ptop,Ppostbottom): #kg/kg, kPa, K,  hPa, hPa
    import numpy as np
    # Solve for LCL Temperature
    Tc = Temp_LCL_Vapor(P0, T0, w)
    k = 287/1004
    E = 287/461
    # Solve for LCL Pressure
    rc0 = w
    L = 2.5 * 10**6
    Rv = 461
    TLCL = Tc
    PLCL = P0/((T0/Tc)**(1004/287)) * 10 # hPa
    # Solve for mixing ratio at top

    rc = np.arange(0,1,0.00001)

    Ttheta600 = TLCL / ((PLCL/Ptop)**k)
    T = Ttheta600 - (rc - rc0) * (L/1004)
    P = (rc0 * np.exp((-L/Rv)*((1/T) - (1/TLCL))) * PLCL) / rc
       
    closestP = find_closest(P, Ptop)
    Ploc = np.where(P == closestP)

    rctop = rc[Ploc]
    rctopgkg = rctop * 1000

    # Find temperature at the top
    Ttop = Ttheta600 - (rctop - rc0) * (L/1004)

    # Find temperature at bottom 

    Tpostbottom = Ttop / ((Ptop/Ppostbottom)**k)

    # Mixing ratio at top = mixing ratio bottom after hill
    wpostbottom = rctopgkg
    wpostbottomkgkg = wpostbottom / 1000

    # Find RH at bottom after hill

    espostbottom = CC_eqn(Tpostbottom)
    e = (wpostbottomkgkg * Ppostbottom)/E

    RHpostbottom = e/espostbottom
    
    return [rctop,Ttop,Tpostbottom,wpostbottomkgkg,RHpostbottom]