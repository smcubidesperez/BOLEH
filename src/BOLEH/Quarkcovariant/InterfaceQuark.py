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
    YTDelta = results["YTDelta"]
    YDU = results["YDU"]
    YDD = results["YDD"]
    y_np = np.asarray(results["y_newphys"])
    
    tr_TDelta = np.array([np.trace(m) for m in YTDelta])
    tr_DU = np.array([np.trace(m) for m in YDU])
    tr_DD = np.array([np.trace(m) for m in YDD])
    YBL =results["YBL"]
    
    print("Final B-L Asymmetry")
    print(r'$Y_{B-L}=$',np.abs(YBL[-1]))

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
        base_name = "Quark_plot"
    else:
        base_name = filename.replace('.h5', '')
        base_name = base_name.replace('at', '')       
        base_name = "_".join(base_name.split())       
        base_name = base_name.strip('_')
    file_bl = base_name + "_BL.png"
    plt.savefig(file_bl, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Plot saved as: {file_bl}")

    
    plt.figure(figsize=(8, 5))
    plt.plot(
        z, np.abs(tr_TDelta),
        lw=2.2, color='tab:blue', ls='-',
        label=r'$\mathrm{Tr}\,Y_{\tilde{\Delta}}$'
    )
    
    plt.plot(
        z, np.abs(tr_DU),
        lw=2.2, color='tab:orange', ls='--',
        label=r'$\mathrm{Tr}\,Y_{\Delta U}$'
    )
    plt.plot(
        z, np.abs(tr_DD),
        lw=2.2, color='tab:green', ls='--',
        label=r'$\mathrm{Tr}\,Y_{\Delta D}$'
    )
    
   
    
    
    title1 = "Quark Sector"
    plt.xlabel('z')
    plt.ylabel(r'$|\rm{Tr}(Y_{\Delta i})|$')
    plt.xlim(z[0],z[-1])
    plt.ylim(1e-15,1e-7)
    plt.xscale('log')
    plt.yscale('log')
    plt.minorticks_on()
    plt.title(title1)
    plt.minorticks_on()
    plt.tick_params(which='both', top=True, right=True, direction='in')
    plt.legend()
    
    file_ferm = base_name + "_Ferm.png"
    plt.savefig(file_ferm, dpi=300, bbox_inches='tight')  
    plt.show()
    print(f" Plot saved as: {file_ferm}")

    
    
    


def SolveBEQuark(z_span,
              y0=None, 
              contributions=None,
              params_sm=None,
              background_funcs=None,
              filename=None,
              rtol=None,
              atol=None):  
    start_time = time.time()
    if params_sm is None:
        params_sm = params_def.copy()
    if rtol is None: rtol = 1e-6
    if atol is None: atol = 1e-9

    def get_newphys_size(contributions):

        nvars = 0
    
        if contributions and "NEW_DEGREES_OF_FREEDOM" in contributions:
            for func in contributions["NEW_DEGREES_OF_FREEDOM"]:
                nvars += func.nvars
    
        return nvars
    
    
    n_newphys = get_newphys_size(contributions)

    n_SM = 27 
    
    if y0 is None:
        y0 = np.zeros(n_SM + n_newphys, dtype=complex)
    
    ode, jac, _ = BE_RHSQuark(y0_total=y0,
                           contributions=contributions,
                           params_sm=params_sm,
                           background_funcs=background_funcs)
                           
    results = BESolverQuark(z_span=z_span, y0=y0, ode_func=ode, jac_func=jac, rtol=rtol, atol=atol)
    
    results["YBL"] = np.array([np.trace(m).real for m in results["YTDelta"]])+1/3*np.array([np.trace(m).real for m in results["YDU"]]) + 1/3*np.array([np.trace(m).real for m in results["YDD"]])
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
    # ---------------------------------------------
    #PlotResultsQuark(results, filename)
    return results
    

