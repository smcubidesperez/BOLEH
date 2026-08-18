#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .physicsL import params_def
import matplotlib.pyplot as plt
import numpy as np
import h5py
from .physicsL import * 
from .solverL import * 
import time
def PlotResultsLE(results, filename):
    z = results["z"]
    #YTDelta = results["YTDelta"]
    #YDE = results["YDE"]
    #y_np = np.asarray(results["y_newphys"])
    
    #tr_TDelta = np.array([np.trace(m) for m in YTDelta])
    #tr_DE = np.array([np.trace(m) for m in YDE])
    YBL =results["YBL"]
    
    #print("Final B-L Asymmetry")
    #print(r'$Y_{B-L}=$',np.abs(YBL[-1]))

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "figure.dpi": 150,
    })
    
    plt.figure(figsize=(8, 5))
    
    
    plt.plot(
        z,
        np.abs(YBL),
        
        label=r'$|Y_{B-L}|$',
        color='red'
    )
    title="B-L Asymmetry"
        
    plt.xlim(z[0],z[-1])
    plt.ylim(1e-15,1e-7)
    plt.xlabel(r'$z$')
    plt.title(title)
    plt.xscale('log')
    plt.yscale('log')
    plt.minorticks_on()
    plt.tick_params(which='both', top=True, right=True, direction='in')
    plt.legend()
    if filename is None:
        base_name = "lepto_plot"
    else:
        base_name = filename.replace('.h5', '')
        base_name = base_name.replace('at', '')       
        base_name = "_".join(base_name.split())       
        base_name = base_name.strip('_')
    file_bl = base_name + "_BL.png"
    plt.savefig(file_bl, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Plot saved as: {file_bl}")

    
    #plt.figure(figsize=(8, 5))
    #plt.plot(
    #    z, np.abs(tr_TDelta),
    #    lw=2.2, color='tab:blue', ls='-',
    #    label=r'$\mathrm{Tr}\,Y_{\Delta L}$'
    #)
    
    #plt.plot(
    #    z, np.abs(tr_DE),
    #    lw=2.2, color='tab:orange', ls='--',
    #    label=r'$\mathrm{Tr}\,Y_{\Delta E}$'
    #)
    
   
    
    
    #title1 = "Lepton Sector"
    #plt.xlabel('z')
    #plt.ylabel(r'$|\rm{Tr}(Y_{\Delta i})|$')
    #plt.xlim(z[0],z[-1])
    #plt.ylim(1e-15,1e-7)
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.minorticks_on()
    #plt.title(title1)
    #plt.minorticks_on()
    #plt.tick_params(which='both', top=True, right=True, direction='in')
    #plt.legend()
    
    #file_ferm = base_name + "_Ferm.png"
    #plt.savefig(file_ferm, dpi=300, bbox_inches='tight')  
    #plt.show()
    #print(f" Plot saved as: {file_ferm}")

    
    
    


def SolveBELE(z_span,
              ynew0=None, 
              contributions=None,
              params_sm=None,
              background_funcs=None,
              filename=None,
              rtol=None,
              atol=None,
              plot=False
              ):  
    if params_sm is None:
        params_sm = params_def.copy()

    if rtol is None: rtol = 1e-6
    if atol is None: atol = 1e-9

    if ynew0 is None:
        raise ValueError("You forgot to define your initial conditions!")
    y0SM = np.zeros(18) 
    
    y0=np.concatenate((y0SM , ynew0),dtype=complex)
    ode, jac, _ = BE_RHSLE(y0_total=y0,
                           contributions=contributions,
                           params_sm=params_sm,
                           background_funcs=background_funcs)
                           
    results = BESolverLepto(z_span=z_span, y0=y0, ode_func=ode, jac_func=jac, rtol=rtol, atol=atol)
    YBL = np.array([np.trace(m).real for m in results["YTDelta"]])-np.array([np.trace(m).real for m in results["YDE"]])
    results["YBL"] = YBL
    # ----------------------------------------

    if filename is not None:
        if not filename.endswith('.h5'): filename += '.h5'
        with h5py.File(filename, 'w') as f:
            f.create_dataset('z', data=np.asarray(results["z"]))
            f.create_dataset('YBL', data=np.asarray(results["YBL"])) 
            f.create_dataset('YTDelta', data=np.asarray(results["YTDelta"]))
            f.create_dataset('YDE', data=np.asarray(results["YDE"]))
            f.create_dataset('y_newphys', data=np.asarray(results["y_newphys"]))
            
        print(f"✅ Data saved successfully to HDF5: {filename}")
    print(fr"Final $Y_B$ (effective Lepton-flavor covariant formalism): {np.abs(YBL[-1]):.4e}")

    if plot:
        PlotResultsLE(results, filename)
    return results
    

