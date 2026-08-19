#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from .physicsQuark import params_def
import matplotlib.pyplot as plt
import numpy as np
import h5py
from .physicsQuark import * 
from .solverQuark import * 
import time
def PlotResultsQuark(results, filename):
    z = results["z"]
    #YTDelta = results["YTDelta"]
    #YDU = results["YDU"]
    #YDD = results["YDD"]
    #y_np = np.asarray(results["y_newphys"])
    
    #tr_TDelta = np.array([np.trace(m) for m in YTDelta])
    #tr_DU = np.array([np.trace(m) for m in YDU])
    #tr_DD = np.array([np.trace(m) for m in YDD])
    YBL =results["YBL"]
    
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "DejaVu Serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 13,
        "figure.dpi": 150,
    })
      
    #Determine maximum and minimum values of |Y_{B-L}| for the plot
    ymax = 10*np.max(np.abs(YBL))
    ymin = 1e-10*np.max(np.abs(YBL))
    
    title="Quark-flavor-covariant formalism"
    
    plt.figure(figsize=(8, 5))
    plt.plot(z, np.abs(YBL), label=r'$|Y_{B-L}|$', color='red')
    plt.xlim(z[0],z[-1])
    plt.ylim(ymin,ymax)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$Y_{B-L}$')
    plt.title(title)
    plt.xscale('log')
    plt.yscale('log')
    plt.minorticks_on()
    plt.tick_params(which='both', top=True, right=True, direction='in')
    plt.grid()
    plt.tight_layout()
    #plt.legend()
    
    if filename is None:
        base_name = "effQ_plot"
    else:
        base_name = filename.replace('.h5', '')
        base_name = base_name.replace('at', '')       
        base_name = "_".join(base_name.split())       
        base_name = base_name.strip('_')   
    
    
    file_bl = base_name + "_BL.png"
    plt.savefig(file_bl, dpi=300, bbox_inches='tight')
    plt.show()


    
    
    


def SolveBEQuark(z_span,
              ynew0=None, 
              contributions=None,
              params_sm=None,
              background_funcs=None,
              filename=None,
              rtol=None,
              atol=None,
              plot=False):  
    start_time = time.time()
    if params_sm is None:
        params_sm = params_def.copy()
    if rtol is None: rtol = 1e-6
    if atol is None: atol = 1e-9
    if ynew0 is None:
        raise ValueError("You forgot to define your initial conditions!")

    y0SM = np.zeros(27) 
    
    y0=np.concatenate((y0SM , ynew0),dtype=complex)
    ode, jac, _ = BE_RHSQuark(y0_total=y0,
                           contributions=contributions,
                           params_sm=params_sm,
                           background_funcs=background_funcs)
                           
    results = BESolverQuark(z_span=z_span, y0=y0, ode_func=ode, jac_func=jac, rtol=rtol, atol=atol)
    YBL = np.array([np.trace(m).real for m in results["YTDelta"]])+1/3*np.array([np.trace(m).real for m in results["YDU"]]) + 1/3*np.array([np.trace(m).real for m in results["YDD"]])
    
    results["YBL"] = YBL
    # ----------------------------------------

    if filename is not None:
        if not filename.endswith('.h5'): filename += 'Quark.h5'
        with h5py.File(filename, 'w') as f:
            f.create_dataset('z', data=np.asarray(results["z"]))
            f.create_dataset('YBL', data=np.asarray(results["YBL"])) 
            f.create_dataset('YTDelta', data=np.asarray(results["YTDelta"]))
            f.create_dataset('YDU', data=np.asarray(results["YDU"]))
            f.create_dataset('YDD', data=np.asarray(results["YDD"]))
            f.create_dataset('y_newphys', data=np.asarray(results["y_newphys"]))
            
        print(f"✅ Data saved successfully to HDF5: {filename}")
        
    print("--- Effective quark-flavor-covariant formalism ---")
    print(fr"Final Y_(B-L): {YBL[-1]:.4e}")
    print(fr"Final Y_B: {0.315*YBL[-1]:.4e}")
    print(fr"Final eta_B: {7.039*0.315*YBL[-1]:.4e}")

    if plot:
        PlotResultsQuark(results, filename)
    return results
    

