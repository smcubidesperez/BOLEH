#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from scipy.integrate import solve_ivp
import time
import sympy as sp

from . import physicsSM as pB

def BESolver(z_span, y0, ode_func, jac_func, rtol=1e-7, atol=1e-10):
    z_init, z_final = z_span
    z_eval = np.logspace(np.log10(z_init), np.log10(z_final), 2000)



    start_time = time.time()

    sol = solve_ivp(
        ode_func, [z_eval[0], z_eval[-1]], y0,
        t_eval=z_eval, method="BDF", jac=jac_func, rtol=rtol, atol=atol
    )
    z_sol=sol.t

    print(f" Complete integration {time.time() - start_time:.4f} seconds.")


    step = 9
    def rebuild_matrix(v):
        return np.array([
            [v[0],           v[3] + 1j*v[4], v[5] + 1j*v[6]],
            [v[3] - 1j*v[4], v[1],           v[7] + 1j*v[8]],
            [v[5] - 1j*v[6], v[7] - 1j*v[8], v[2]]
        ], dtype=np.complex128)

    YDL = [rebuild_matrix(y[0:9]) for y in sol.y.T]
    YDE = [rebuild_matrix(y[9:18]) for y in sol.y.T]
    YDQ = [rebuild_matrix(y[18:27]) for y in sol.y.T]
    YDU = [rebuild_matrix(y[27:36]) for y in sol.y.T]
    YDD = [rebuild_matrix(y[36:45]) for y in sol.y.T]
    YN = [y[45:] for y in sol.y.T]
   
    results = {
        "z": z_sol,  
        "YDQ": YDQ, "YDU": YDU, "YDD": YDD, "YDL": YDL, "YDE": YDE, "y_newphys": YN
    }
    
    return results

