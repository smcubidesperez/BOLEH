#!/usr/bin/env python
# coding: utf-8

import sympy as sp
import numpy as np
from scipy.special import kn
from ..config import config
import time
# GLOBAL CONSTANTS AND PARAMETERS

gH, gQ, gU, gD =  2, 6, 3, 3
ZetaH, ZetaQ, ZetaU, ZetaD = 2, 1, 1, 1
geff = 106.75
Ynor = 15 / (8 * np.pi**2 * geff)
mpl = 1.22e19  # GeV
n = 3

H_BASE = 1.66 * np.sqrt(geff) / mpl
S_BASE = 2 * (np.pi**2 / 45) * geff

G2_CONST = 0.55
ALPHA2_CONST = G2_CONST**2 / (4 * np.pi)
GAMMA2_PRE = (13.7 + 4.49 * np.log(1.35 / G2_CONST)) * ALPHA2_CONST**5

G3_CONST = 0.61
ALPHA3_CONST = G3_CONST**2 / (4 * np.pi)
GAMMA3_PRE = (95.1 + 31.3 * np.log(1.41 / G3_CONST)) * ALPHA3_CONST**5
# CKM y acoplamientos por defecto
theta12, theta23, theta13, delta = 0.227, 4.65e-2, 4.11e-3, 1.139j
V23 = np.array([[1, 0, 0], [0, np.cos(theta23), np.sin(theta23)], [0, -np.sin(theta23), np.cos(theta23)]], dtype=np.complex128)
V13 = np.array([[np.cos(theta13), 0, np.sin(theta13)*np.exp(-delta)], [0, 1, 0], [-np.sin(theta13)*np.exp(delta), 0, np.cos(theta13)]], dtype=np.complex128)
V12 = np.array([[np.cos(theta12), np.sin(theta12), 0], [-np.sin(theta12), np.cos(theta12), 0], [0, 0, 1]], dtype=np.complex128)
VCKM = V23 @ V13 @ V12

yU_def = np.array([[4.39e-6, 0, 0], [0, 1.98e-3, 0], [0, 0, 0.4454]], dtype=np.complex128)
yDdiag = np.array([[0.97e-5, 0, 0], [0, 1.72e-4, 0], [0, 0, 0.719e-2]], dtype=np.complex128)
yD_def = yDdiag @ np.conj(VCKM.T)
params_def = {"yU": yU_def,
             "yD": yD_def}


# Thermal functions

def Hubble(z):
    Mref1 = config["Mref"]
    return H_BASE * (Mref1 / z)**2

def s(z):
    Mref1 = config["Mref"]
    return S_BASE * (Mref1 / z)**3

def _common_factor(z):
    Mref1 = config["Mref"]
    T = Mref1 / z
    return 1.0 / ((S_BASE * T**3) * (H_BASE * T**2) * z)

def gammaU(z):
    Mref1 = config["Mref"]
    T = Mref1 / z
    coeffU = -8.2e-6*(np.log10(T))**3 + 3e-4*(np.log10(T))**2 -4e-3*(np.log10(T)) + 3e-2
    return coeffU * T**4 * _common_factor(z)

def gammaD(z):
    Mref1 = config["Mref"]
    T = Mref1 / z
    coeffD = -8.2e-6*(np.log10(T))**3 + 3e-4*(np.log10(T))**2 -4e-3*(np.log10(T)) + 3e-2
    return coeffD * T**4 * _common_factor(z)
    
def kv_safe(nu, z_val):
    val = kn(nu, z_val)
    return np.where(val == 0.0, np.finfo(np.float64).tiny, val)
def gammaQCD(z):
    Mref1 = config["Mref"]
    T = Mref1 / z
    return (GAMMA3_PRE * T**4) * _common_factor(z)
    
def build_hermitian_matrix_sym(vars_9):
    d0, d1, d2, r01, i01, r02, i02, r12, i12 = vars_9
    return sp.Matrix([
        [d0, r01 + sp.I * i01, r02 + sp.I * i02],
        [r01 - sp.I * i01, d1, r12 + sp.I * i12],
        [r02 - sp.I * i02, r12 - sp.I * i12, d2]
    ])

def cq1(z):
    Mref1 = config["Mref"]
    T_inv = z / Mref1
    
    if hasattr(z, 'free_symbols') or isinstance(z, sp.Basic):
        exp = lambda t: sp.exp(-t * T_inv)
    else:
        
        exp = lambda t: np.exp(-t * T_inv)
    return (-3/4*(1-exp(2.3e12)) - (-3/4 + 24/33)*(1-exp(4e11)) -
            (-24/33+15/21)*(1-exp(1e9)) - (-15/21+12/17)*(1-exp(3e4)))
    
def cq2(z):
    Mref1 = config["Mref"]
    T_inv = z / Mref1
    
    if hasattr(z, 'free_symbols') or isinstance(z, sp.Basic):
        exp = lambda t: sp.exp(-t * T_inv)
    else:
        
        exp = lambda t: np.exp(-t * T_inv)
    return (1/33*(1-exp(4e11)) - (1/33 - 1/21)*(1-exp(1e9)) -
            (1/21-1/17)*(1-exp(3e4)))
def ch1(z):
    Mref1 = config["Mref"]
    T_inv = z / Mref1
   
    if hasattr(z, 'free_symbols') or isinstance(z, sp.Basic):
        exp = lambda t: sp.exp(-t * T_inv)
    else:
        
        exp = lambda t: np.exp(-t * T_inv)
    return (-1-(-1+10/11)*(1-exp(4e11)) - (-10/11 + 18/21)*(1-exp(1e9)) -
            (-18/21+42/51)*(1-exp(3e4)))
def ch2(z):
    Mref1 = config["Mref"]
    T_inv = z / Mref1
   
    if hasattr(z, 'free_symbols') or isinstance(z, sp.Basic):
        exp = lambda t: sp.exp(-t * T_inv)
    else:
       
        exp = lambda t: np.exp(-t * T_inv)
    return (-2/3-(-2/3+6/11)*(1-exp(4e11)) - (-6/11 + 10/21)*(1-exp(1e9)) -
            (-10/21+22/51)*(1-exp(3e4)))

# Boltzmann Equations 

def BoltzmannQuark(z, y_vec, contributions=None, params_sm=None):
    inv_gQZQ = 1.0 / (gQ * ZetaQ)
    inv_gUzu = 1.0 / (gU * ZetaU)
    inv_gDZz = 1.0 / (gD * ZetaD)
    inv_gHZH = 1.0 / (gH * ZetaH)
    inv_Ynor = 1.0 / Ynor

    p = params_def if params_sm is None else params_sm
    
    if isinstance(y_vec[0], sp.Symbol): 
        YTDelta = build_hermitian_matrix_sym(y_vec[0:9])
        YDU = build_hermitian_matrix_sym(y_vec[9:18])
        YDD = build_hermitian_matrix_sym(y_vec[18:27])
        y_newphys = y_vec[27:]
        eye_func = sp.eye
    else:
        def build_hermitian_matrix_num(v):
            return np.array([[v[0], v[3]+1j*v[4], v[5]+1j*v[6]],
                             [v[3]-1j*v[4], v[1], v[7]+1j*v[8]],
                             [v[5]-1j*v[6], v[7]-1j*v[8], v[2]]], dtype=np.complex128)
        YTDelta = build_hermitian_matrix_num(y_vec[0:9])
        YDU = build_hermitian_matrix_num(y_vec[9:18])
        YDD = build_hermitian_matrix_num(y_vec[18:27])
        y_newphys = y_vec[27:]
        eye_func = np.eye

    def get_param(param, z):
        return param(z) if callable(param) else param
    yU = get_param(p["yU"], z)
    yD = get_param(p["yD"], z)

    def dagger(A):
        if isinstance(A, sp.Matrix):
            return A.H
        return A.T.conj()
        
    yUT, yDT = dagger(yU), dagger(yD)
    yDTyD, yDyDT = yDT @ yD, yD @ yDT
    yUTyU, yUyUT = yUT @ yU, yU @ yUT

    trYDD, trYDU, trYTDelta = YDD.trace(), YDU.trace(), YTDelta.trace()
    
    YDQ = 3*YTDelta +cq1(z)* trYTDelta*eye_func(3) + cq2(z)* (2*trYDU-trYDD) * eye_func(3)
    YDH = ch1(z) * trYTDelta * eye_func(3) + ch2(z) * (2*trYDU-trYDD) * eye_func(3)

    term_YDU_scaled = YDU * inv_gUzu
    term_YDD_scaled = YDD * inv_gDZz
    term_YDQ_scaled = YDQ * inv_gQZQ
    trYDQ = YDQ.trace()
    cqcd_val = (gammaQCD(z) / 6.0 * inv_Ynor) * (2.0 * trYDQ * inv_gQZQ - trYDU * inv_gUzu - trYDD * inv_gDZz)
    CQCD = cqcd_val * eye_func(3)

    if contributions and "HYPERCHARGES" in contributions:
        hypercharge_dict = contributions["HYPERCHARGES"]
        for idx, Y_val in hypercharge_dict.items():
            y_field_val = y_newphys[idx]
            contrib_field = y_field_val.trace() if hasattr(y_field_val, 'trace') else y_field_val
            YDQ += cq2(z) * (3.0 * Y_val * contrib_field) * eye_func(3)
            YDH += ch2(z) * (3.0 * Y_val * contrib_field) * eye_func(3)

    
    gU_step = gammaU(z) * inv_Ynor
    gD_step = gammaD(z) * inv_Ynor
    rhs_newphys = []

    rhsLepto = {
        "YTDelta": 1/3 * (- 2.0*CQCD - (gU_step * 0.5) * (yUTyU @ term_YDQ_scaled + term_YDQ_scaled @ yUTyU) - gU_step * (yUTyU @ (YDH * inv_gHZH)) + gU_step * (yUT @ (YDU * inv_gUzu) @ yU) - (gD_step * 0.5) * (yDTyD @ term_YDQ_scaled + term_YDQ_scaled @ yDTyD) + gD_step * (yDTyD @ (YDH * inv_gHZH)) + gD_step * (yDT @ (YDD * inv_gDZz) @ yD)),
        "YDU": (CQCD - (gU_step * 0.5) * ((yU@yUT) @ (YDU * inv_gUzu) + (YDU * inv_gUzu) @ (yU@yUT)) + gU_step * (yU @ yUT @ (YDH * inv_gHZH)) + gU_step * (yU @ term_YDQ_scaled @ yUT)),
        "YDD": (CQCD - (gD_step * 0.5) * ((yD@yDT) @ (YDD * inv_gDZz) + (YDD * inv_gDZz) @ (yD@yDT)) - gD_step * (yD @ yDT @ (YDH * inv_gHZH)) + gD_step * (yD @ term_YDQ_scaled @ yDT))
    }
    if isinstance(y_vec[0], sp.Symbol):
        zero_matrix = sp.zeros(3, 3)
    else:
        zero_matrix = np.zeros((3, 3), dtype=np.complex128)
    
    
    YDL = zero_matrix
    YDE = zero_matrix
    if contributions:
        for channel, new_functions in contributions.items():
            for func in new_functions:
                if channel == "NEW_DEGREES_OF_FREEDOM":
                    rhs_newphys.append(func(z,y_newphys, YDL, YDE, YDH, YDQ, YDU, YDD))
                elif channel in rhsLepto:
                    rhsLepto[channel] += func(z,y_newphys, YDL, YDE, YDH, YDQ, YDU, YDD)

    f_list = []
    for name in ["YTDelta", "YDU", "YDD"]:
        dm = rhsLepto[name]
        f_list.extend([
            sp.re(dm[0,0]) if isinstance(dm, sp.Matrix) else np.real(dm[0,0]),
            sp.re(dm[1,1]) if isinstance(dm, sp.Matrix) else np.real(dm[1,1]),
            sp.re(dm[2,2]) if isinstance(dm, sp.Matrix) else np.real(dm[2,2]),
            sp.re(dm[0,1]) if isinstance(dm, sp.Matrix) else np.real(dm[0,1]),
            sp.im(dm[0,1]) if isinstance(dm, sp.Matrix) else np.imag(dm[0,1]),
            sp.re(dm[0,2]) if isinstance(dm, sp.Matrix) else np.real(dm[0,2]),
            sp.im(dm[0,2]) if isinstance(dm, sp.Matrix) else np.imag(dm[0,2]),
            sp.re(dm[1,2]) if isinstance(dm, sp.Matrix) else np.real(dm[1,2]),
            sp.im(dm[1,2]) if isinstance(dm, sp.Matrix) else np.imag(dm[1,2])
        ])

    if len(rhs_newphys) > 0:
        for item in rhs_newphys:
            if hasattr(item, '__iter__') or isinstance(item, sp.Matrix):
                f_list.extend(list(item))
            else:
                f_list.append(item)

    return f_list if isinstance(y_vec[0], sp.Symbol) else np.array(f_list)


def BE_RHSQuark(y0_total, contributions=None, params_sm=None, background_funcs=None):
    import sys
    main_mod = sys.modules['__main__']
    if background_funcs is None:
        background_funcs = []

    total_vars = len(y0_total)

    z_sym = sp.symbols('z', real=True, positive=True)
    y_symbols = sp.symbols(f'y0:{total_vars}', real=True)

    p = params_def if params_sm is None else params_sm
    params_sym = p.copy()  

    dict_translate = {}

    orig_yD = params_sym.get("yD", None)
    orig_yU = params_sym.get("yU", None)

   
    if callable(orig_yD):
        params_sym["yD"] = yD_mock
    if callable(orig_yU):
        params_sym["yU"] = yU_mock
    sm_funcs = ['gammaU', 'gammaD', 'gammaQCD']
    sym_masks = {name: sp.Function(f"{name}_sym") for name in sm_funcs}
    def gammaQCD_mock(z):
        return sym_masks['gammaQCD'](z)
    
    def gammaU_mock(z):
        return sym_masks['gammaU'](z)
    
    def gammaD_mock(z):
        return sym_masks['gammaD'](z)
    def yD_mock(z):
        return sp.Matrix([
            [sp.Function("yD11_sym")(z), sp.Function("yD12_sym")(z), sp.Function("yD13_sym")(z)],
            [sp.Function("yD21_sym")(z), sp.Function("yD22_sym")(z), sp.Function("yD23_sym")(z)],
            [sp.Function("yD31_sym")(z), sp.Function("yD32_sym")(z), sp.Function("yD33_sym")(z)],
        ])

    def yU_mock(z):
        return sp.Matrix([
            [sp.Function("yU11_sym")(z), sp.Function("yU12_sym")(z), sp.Function("yU13_sym")(z)],
            [sp.Function("yU21_sym")(z), sp.Function("yU22_sym")(z), sp.Function("yU23_sym")(z)],
            [sp.Function("yU31_sym")(z), sp.Function("yU32_sym")(z), sp.Function("yU33_sym")(z)],
        ])

    orig_np = {}

    for name in background_funcs:
        if hasattr(main_mod, name):
            orig_np[name] = getattr(main_mod, name)
    
            sympy_func_obj = sp.Function(f"{name}_sym")
    
            setattr(
                main_mod,
                name,
                lambda z, i=None, obj=sympy_func_obj:
                    obj(z, i) if i is not None else obj(z)
            )
    global gammaQCD, gammaU, gammaD
   
    orig_sm = (
        gammaU,
        gammaD,
        gammaQCD
    )

    gammaU = gammaU_mock
    gammaD = gammaD_mock
    gammaQCD = gammaQCD_mock
    
    try:
        rhs_simbolic = BoltzmannQuark(z_sym, list(y_symbols), contributions, params_sym)
    finally:
        
        gammaU, gammaD, gammaQCD = orig_sm
        if orig_yD is not None:
            params_sym["yD"] = orig_yD
        if orig_yU is not None:
            params_sym["yU"] = orig_yU
        for name, orig_func in orig_np.items():
            setattr(main_mod, name, orig_func)

    rhs_simbolic_pure = []
    for expr in rhs_simbolic:
        if hasattr(expr, 'as_explicit'):  
            rhs_simbolic_pure.append(expr.as_explicit())
        elif hasattr(expr, '__getitem__') and not isinstance(expr, (sp.Matrix, sp.Basic)):
            rhs_simbolic_pure.append(expr[0])
        else:
            rhs_simbolic_pure.append(sp.sympify(expr))
            
    f_matrix = sp.Matrix(rhs_simbolic_pure)
    
    print("--- Analytical Jacobian Matrix ---")
    start_time = time.time()
    J_symbolic = f_matrix.jacobian(y_symbols)

    
    # --- Traducción de funciones simbólicas a numéricas ---
    
    for name, orig_func in orig_np.items():
        dict_translate[f"{name}_sym"] = lambda z, i=None, f=orig_func: (
            f(z)[int(i)] if i is not None else f(z)
        )
    
    
    # --- yD(z), si es función ---
    
    if orig_yD is not None:
        for i in range(3):
            for j in range(3):
                dict_translate[f"yD{i+1}{j+1}_sym"] = (
                    lambda z, ii=i, jj=j, f=orig_yD:
                    f(z)[ii, jj]
                )
    
    
    # --- yU(z), si es función ---
    
    if orig_yU is not None:
        for i in range(3):
            for j in range(3):
                dict_translate[f"yU{i+1}{j+1}_sym"] = (
                    lambda z, ii=i, jj=j, f=orig_yU:
                    f(z)[ii, jj]
                )
    
    
    # --- Funciones mockeadas ---
    
    dict_translate.update({
        'gammaU_sym': lambda z: orig_sm[0](z),
        'gammaD_sym': lambda z: orig_sm[1](z),
        'gammaQCD_sym': lambda z: orig_sm[2](z)
    })
    
    
    # --- Modules para lambdify ---
    
    map_modules = [
        dict_translate,
        'numpy'
    ]
    
    print("--- Lambdify ---")
    
    ode_lambda = sp.lambdify(
        (z_sym, y_symbols),
        f_matrix,
        modules=map_modules,
        cse=True
    )
    
    jac_lambda = sp.lambdify(
        (z_sym, y_symbols),
        J_symbolic,
        modules=map_modules,
        cse=True
    )
    
    
    # --- Funciones numéricas finales ---
    
    def ode_analitic(z, y):
        return np.asarray(
            ode_lambda(z, y),
            dtype=complex
        ).ravel()
    
    
    def jac_analitic(z, y):
        return np.asarray(
            jac_lambda(z, y),
            dtype=complex
        )
    
    print(f"Time of Jacobian Calculation {time.time() - start_time:.4f} seconds.")
    print("End of Jacobian and Ode")
    
    return ode_analitic, jac_analitic, total_vars
