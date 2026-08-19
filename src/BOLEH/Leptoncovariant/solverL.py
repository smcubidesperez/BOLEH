#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.integrate import solve_ivp
import time
import sympy as sp

from . import physicsL as pl

def BESolverLepto(z_span, y0, ode_func, jac_func, rtol, atol):
    z_init, z_final = z_span
    num_z_eval = int((np.log10(z_final) - np.log10(z_init))*100)
    z_eval = np.logspace(np.log10(z_init), np.log10(z_final), num_z_eval)
    
    start_time = time.time()
    
    sol = solve_ivp(
        ode_func, [z_eval[0], z_eval[-1]], y0,
        t_eval=z_eval, method="BDF", jac=jac_func,
        rtol=rtol, atol=atol
    )
    z_sol = sol.t
    
    print(f"Time for numerical integration = {time.time() - start_time:.4f} seconds.")
    
    def rebuild_matrix(v):
        return np.array([
            [v[0],           v[3] + 1j*v[4], v[5] + 1j*v[6]],
            [v[3] - 1j*v[4], v[1],           v[7] + 1j*v[8]],
            [v[5] - 1j*v[6], v[7] - 1j*v[8], v[2]]
        ], dtype=np.complex128)

    YTDelta = [rebuild_matrix(y[0:9]) for y in sol.y.T]
    YDE = [rebuild_matrix(y[9:18]) for y in sol.y.T]
    YN = [y[18:] for y in sol.y.T]
    
    results = {
        "z": z_sol,  
        "YTDelta": YTDelta, "YDE": YDE, "y_newphys": YN
    }
    return results
