#Example of leptogenesis from one right-handed neutrino decay

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from scipy.special import kn, kve
import re

import sys
sys.path.append("../src")
import BOLEH as bl


#Define the parameters for the model
y = np.array([0.01*np.exp(0.1 * 1j), 0.02*np.exp(0.2 * 1j), 0.03*np.exp(0.3 * 1j)])
M = 1.e12 #GeV


#Define reference scale to define z = Mref/T
Mref = M
bl.config["Mref"] = Mref


#Construct flavour projector
yydag = y @ y.conj().T
PP = np.outer(y.conj(), y)/yydag


#Provide a Hermitian CP parameter
ep = (M/1.e12)*np.array([[4.6*1e-8, 1.2*1e-6*np.exp(-1.8 * 1j), 1.2*1e-6*np.exp(-1.9 * 1j)],
               [1.2*1e-7*np.exp(1.8 * 1j), -1.8*1e-6, 2.9*1e-6*np.exp(2.8 * 1j)],
               [1.2*1e-6*np.exp(1.9 * 1j), 2.9*1e-6*np.exp(-2.8 * 1j), -3.8*1e-6]])

#Define functions for the model
#Equilibrium abundance of right-handed neutrino N with mass M
def YNeq(z):
    T = Mref / z
    x = M / T
    YNEQ_PRE = 45 / (2 * np.pi**4 * bl.geff)
    return YNEQ_PRE * x**2 * kn(2, x)

#Decay reaction density of N divided by sHz
def gammaNY(z):
    T = Mref / z
    x = M / T
    GAMMAN_VEC = yydag * M / (8 * np.pi)
   
    ratio = np.where(
        x < 50,
        kve(1, x) / kve(2, x),
        1.0 + 1.5 / x
    )
    gN = GAMMAN_VEC * ratio  / (z * bl.Hubble(z))
    return gN


#Define the washout term
def W(z, y_newphys, YDL, YDE, YDH, YDQ, YDU, YDD):

    term_l = YDL * 1 / (bl.gl*bl.Zetal)
    term_h = YDH * 1 / (bl.gH* bl.ZetaH) 

    WI = gammaNY(z)  * YNeq(z) / bl.Ynor * (0.5 * (PP @ term_l + term_l @ PP) + (PP @ term_h))
        
    return -0.5 * WI  

#Define the source term
def SI(z, y_newphys, YDL, YDE, YDH, YDQ, YDU, YDD):
    YN = y_newphys
    
    Si = ep * gammaNY(z) * (YN[0] - YNeq(z))     
        
    return Si

#Define new Boltzmann equations, one for each new variable
def dYN(z, y_newphys, YDL, YDE, YDH, YDQ, YDU, YDD):
    dYN_list = - gammaNY(z) * (y_newphys[0] - YNeq(z)) 
    return dYN_list

#Specify the range of z to solve
zspan=[1e-3, 1e3]

#Specify the vector of initial conditions for the new variables
#If there are more than one variable, the order of the entries of the vector follows the order
#when specifying "NEW_DEGREES_OF_FREEDOM" below. 
Yini = [YNeq(zspan[0])] #Thermal initial abundance

#The functions to protect are the user-defined functions
funcions_to_protect = ['gammaNY', 'YNeq']

#Solving using complete SM-flavor covariant formalism
#Provide the functions of the new Boltzmann equations in "NEW_DEGREES_OF_FREEDOM"
#Provide the source and washout terms (order does not matter) for one of the SM Boltzmann equations: YDQ, YDU, YDD, YDL, YDE
TypeI_contributions = { 
    "NEW_DEGREES_OF_FREEDOM": [dYN],
    "YDL": [W, SI]
}

#Without specifying rtol and atol, they will be set to rtol = 1e-7 and atol = 1e-10
#Set plot=True for plot of Y_{B-L} versus z. By default, plot=False.
results = bl.SolveBE(zspan, ynew0=Yini,
    contributions=TypeI_contributions, 
    background_funcs=funcions_to_protect , 
    filename= 'Type_I',rtol=1e-8,atol=1e-11
)


#Solving using effective lepton-flavor-covariant formalism
#Provide the functions of the new Boltzmann equations in "NEW_DEGREES_OF_FREEDOM"
#Provide the source and washout terms (order does not matter) for one of the effL Boltzmann equations: YTDelta, YDE
#By default, the source and washout terms enter the YTDelta equation with minus sign
TypeI_contributionsL = { 
    "NEW_DEGREES_OF_FREEDOM": [dYN], 
    "YTDelta": [W, SI]
}

#Without specifying rtol and atol, they will be set to rtol = 1e-7 and atol = 1e-10
#Set plot=True for plot of Y_{B-L} versus z. By default, plot=False.
resultsL = bl.SolveBELE(zspan, ynew0=Yini,
    contributions=TypeI_contributionsL, 
    background_funcs=funcions_to_protect,
    filename= 'Type_IL',rtol=1e-8,atol=1e-11
)


#Make some nice plots
#Read the data
z_L = resultsL["z"]
YBL_L = resultsL["YBL"]
YN = resultsL["y_newphys"]

z_B = results["z"]
YBL_B = results["YBL"]

plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "figure.dpi": 150,
    })

#Plot Y_N
#Determine maximum and minimum values of y for the plot
ymax = 10*np.max(np.abs(YN))
ymin = 1e-10*np.max(np.abs(YN))

plt.xlim(zspan[0], zspan[-1])  
plt.ylim(ymin, ymax)
plt.xlabel(r'$z$')
plt.ylabel(r'$Y_{N}$')  
plt.xscale('log')
plt.yscale('log')
plt.minorticks_on()
plt.tick_params(which='both', top=True, right=True, direction='in')
plt.plot(
    z_B,
    np.abs(YN),
    label=r'$Y_{N}$',
    color='black',
    linestyle='-',
    linewidth=1.5
)
plt.grid()
plt.tight_layout()
plt.show()

#Plot |Y_{B-L}| for complete and effL formalism
#Complete formalism
plt.plot(
    z_B,
    np.abs(YBL_B),
    label=r'$|Y_{B-L}|$',
    color='black',
    linestyle='-',
    linewidth=1.5
)

#effL formalism
plt.plot(
    z_L,
    np.abs(YBL_L),
    label=r'$|Y_{B-L}^{\rm eff}|$',
    color='red',
    linestyle='--',
    linewidth=1.5
)

#Determine maximum and minimum values of y for the plot
ymax = 10*np.max(np.abs(YBL_B,YBL_L))
ymin = 1e-10*np.max(np.abs(YBL_B,YBL_L))

plt.xlim(zspan[0], zspan[-1])  
plt.ylim(ymin, ymax)
plt.xlabel(r'$z$')
plt.xscale('log')
plt.yscale('log')
plt.minorticks_on()
plt.tick_params(which='both', top=True, right=True, direction='in')
plt.legend()  
plt.grid()
plt.tight_layout()
plt.show()


#Plot c_H for complete and effL formalism
z_B = results["z"]
YDL = results["YDL"]
YDE = results["YDE"]
YDQ = results["YDQ"]
YDU = results["YDU"]
YDD = results["YDD"]
YBL = results["YBL"]

tr_DL = np.array([np.trace(m) for m in YDL])
tr_DE = np.array([np.trace(m) for m in YDE])
tr_DQ = np.array([np.trace(m) for m in YDQ])
tr_DU = np.array([np.trace(m) for m in YDU])
tr_DD = np.array([np.trace(m) for m in YDD])

den = (1/3 *(tr_DQ + tr_DU + tr_DD) - tr_DL - 2*tr_DE)
c_H2= -(1/3)*(-tr_DQ -4*tr_DU + 2*tr_DD + 3*tr_DL + 6*tr_DE)/den

plt.plot(
    z_B,
    np.abs(c_H2),
    label=r'$|c_H|$',
    color='black'
)

plt.plot(z_B,
    np.abs(bl.ch(z_B)),
    label=r'$c_H^{\rm eff}$',
    color='red',
    linestyle ='--')

plt.xlim(zspan[0], zspan[-1])  
plt.ylim(1e-2,1e0)
plt.xlabel(r'$z$')
plt.xscale('log')
plt.minorticks_on()
plt.tick_params(which='both', top=True, right=True, direction='in')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

