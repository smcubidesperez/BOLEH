#!/usr/bin/env python
# coding: utf-8

from .physicsSM import params_def
import matplotlib.pyplot as plt
import numpy as np
import h5py  # 
from .physicsSM import * 
from .solverSM import * 
import time
def PlotResults(results, filename):
    z = results["z"]
    YDL = results["YDL"]
    YDE = results["YDE"]
    YDQ = results["YDQ"]
    YDU = results["YDU"]
    YDD = results["YDD"]
    YBL = results["YBL"]
    y_np = np.asarray(results["y_newphys"])
    
    tr_DL = np.array([np.trace(m) for m in YDL])
    tr_DE = np.array([np.trace(m) for m in YDE])
    tr_DQ = np.array([np.trace(m) for m in YDQ])
    tr_DU = np.array([np.trace(m) for m in YDU])
    tr_DD = np.array([np.trace(m) for m in YDD])
    
    
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
    plt.plot(z, np.abs(YBL), label=r'$|Y_{B-L}|$', color='red')
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
        base_name = "baryo_plot"
    else:
        base_name = filename.replace('.h5', '')
        base_name = base_name.replace('at', '')       
        base_name = "_".join(base_name.split())       
        base_name = base_name.strip('_')   
    
    
    file_bl = base_name + "_BL.png"
    plt.savefig(file_bl, dpi=300, bbox_inches='tight')
    plt.show()
    print(f" Plot saved as: {file_bl}")


    plt.rcParams['figure.figsize'] = (6, 5)
    plt.rcParams['figure.dpi'] = 150
    
    
    plt.rcParams['axes.labelsize'] = 20     
    plt.rcParams['xtick.labelsize'] = 19    
    plt.rcParams['ytick.labelsize'] = 19   
    plt.rcParams['legend.fontsize'] = 16
    
    
    plt.plot(
        z, np.abs(tr_DL),
        lw=1.5, color='blue', ls='-',
        label=r'$|\mathrm{Tr}\,Y_{\Delta L}|$'
    )
    
    plt.plot(
        z, np.abs(tr_DE),
        lw=1.5, color='tab:green', ls='--',
        label=r'$|\mathrm{Tr}\,Y_{\Delta E}|$'
    )
    
    plt.plot(
        z, np.abs(tr_DQ),
        lw=1.5, color='tab:red', ls=':',
        label=r'$|\mathrm{Tr}\,Y_{\Delta Q}|$'
    )
    
    plt.plot(
        z, np.abs(tr_DU),
        lw=1.5, color='tab:orange', ls='-.',
        label=r'$|\mathrm{Tr}\,Y_{\Delta U}|$'
    )
    
    plt.plot(
        z, np.abs(tr_DD),
        lw=1.5, color='magenta', ls=(0, (7, 3)),
        label=r'$|\mathrm{Tr}\,Y_{\Delta D}|$'
    )

    
    title1 = "SM Fermions"
    plt.xlabel('z')
    plt.ylabel(r'$|\rm{Tr}(Y_{\Delta i})|$')
    plt.xlim(z[0],z[-1])
    plt.ylim(1e-15,1e-7)
    plt.xscale('log')
    plt.yscale('log')
    plt.minorticks_on()
    plt.title(title1)
    plt.tick_params(which='both', top=True, right=True, direction='in')
    plt.legend()
    file_ferm = base_name + "_Ferm.png"
    plt.savefig(file_ferm, dpi=300, bbox_inches='tight') 
    plt.show()
    print(f" Plot saved as: {file_ferm}")
    

def SolveBE(z_span,
            y0=None, 
            contributions=None,
            params_sm=None,
            background_funcs=None,
            filename=None,
            rtol=None,
            atol=None):  
    if params_sm is None:
        params_sm = params_def.copy()
    if rtol is None: rtol = 1e-7
    if atol is None: atol = 1e-10
    
    def get_newphys_size(contributions):

        nvars = 0
    
        if contributions and "NEW_DEGREES_OF_FREEDOM" in contributions:
            for func in contributions["NEW_DEGREES_OF_FREEDOM"]:
                nvars += func.nvars
    
        return nvars
    
    
    n_newphys = get_newphys_size(contributions)

    n_SM = 45
    if y0 is None:
        y0 = np.zeros(n_SM + n_newphys, dtype=complex)
        
    ode, jac, _ = BE_RHS(y0_total=y0,
                         contributions=contributions,
                         params_sm=params_sm,
                         background_funcs=background_funcs)
                         
    results = BESolver(z_span=z_span, y0=y0, ode_func=ode, jac_func=jac, rtol=rtol, atol=atol)
    
    
    tr_DL = np.array([np.trace(m) for m in results["YDL"]])
    tr_DE = np.array([np.trace(m) for m in results["YDE"]])
    tr_DQ = np.array([np.trace(m) for m in results["YDQ"]])
    tr_DU = np.array([np.trace(m) for m in results["YDU"]])
    tr_DD = np.array([np.trace(m) for m in results["YDD"]])
    
    results["YBL"] = (1/3 * (tr_DQ + tr_DU + tr_DD)) - tr_DL - tr_DE
    # -----------------------------

    if filename is not None:
        if not filename.endswith('.h5'): filename += '.h5'
        with h5py.File(filename, 'w') as f:
            f.create_dataset('z', data=np.asarray(results["z"]))
            f.create_dataset('YBL', data=np.asarray(results["YBL"])) 
            f.create_dataset('YDL', data=np.asarray(results["YDL"]))
            f.create_dataset('YDE', data=np.asarray(results["YDE"]))
            f.create_dataset('YDQ', data=np.asarray(results["YDQ"]))
            f.create_dataset('YDU', data=np.asarray(results["YDU"]))
            f.create_dataset('YDD', data=np.asarray(results["YDD"]))
            f.create_dataset('y_newphys', data=np.asarray(results["y_newphys"]))
            
        print(f"✅ Data saved successfully to HDF5: {filename}")
    # ---------------------------------------------
    #PlotResults(results, filename)
    return results