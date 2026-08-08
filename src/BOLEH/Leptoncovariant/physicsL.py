#!/usr/bin/env python
# coding: utf-8

import sympy as sp
import numpy as np
from scipy.special import kn
from ..config import config
from scipy.sparse import csc_matrix
# GLOBAL CONSTANTS AND PARAMETERS

gl, gH, gE = 2, 2, 1
Zetal, ZetaH, ZetaE = 1, 2, 1
geff = 106.75
Ynor = 15 / (8 * np.pi**2 * geff)
mpl = 1.22e19  # GeV
n = 3

H_BASE = 1.66 * np.sqrt(geff) / mpl
S_BASE = 2 * (np.pi**2 / 45) * geff

G2_CONST = 0.55
ALPHA2_CONST = G2_CONST**2 / (4 * np.pi)
GAMMA2_PRE = (13.7 + 4.49 * np.log(1.35 / G2_CONST)) * ALPHA2_CONST**5

# CKM y acoplamientos por defecto
theta12, theta23, theta13, delta = 0.227, 4.65e-2, 4.11e-3, 1.139j
V23 = np.array([[1, 0, 0], [0, np.cos(theta23), np.sin(theta23)], [0, -np.sin(theta23), np.cos(theta23)]], dtype=np.complex128)
V13 = np.array([[np.cos(theta13), 0, np.sin(theta13)*np.exp(-delta)], [0, 1, 0], [-np.sin(theta13)*np.exp(delta), 0, np.cos(theta13)]], dtype=np.complex128)
V12 = np.array([[np.cos(theta12), np.sin(theta12), 0], [-np.sin(theta12), np.cos(theta12), 0], [0, 0, 1]], dtype=np.complex128)
VCKM = V23 @ V13 @ V12

yE_def = np.array([[2.8e-6, 0, 0], [0, 5.9e-4, 0], [0, 0, 1e-2]], dtype=np.complex128)
params_def = {"yE": yE_def}



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

def gammaE(z):
    Mref1 = config["Mref"]
    T = Mref1 / z
    coeffE = -5e-7*(np.log10(T))**3 + 2.1e-5*(np.log10(T))**2 -3.2e-4*(np.log10(T)) + 6.6e-3
    return coeffE * T**4 * _common_factor(z)

def kv_safe(nu, z_val):
    val = kn(nu, z_val)
    return np.where(val == 0.0, np.finfo(np.float64).tiny, val)

def build_hermitian_matrix_sym(vars_9):
    d0, d1, d2, r01, i01, r02, i02, r12, i12 = vars_9
    return sp.Matrix([
        [d0, r01 + sp.I * i01, r02 + sp.I * i02],
        [r01 - sp.I * i01, d1, r12 + sp.I * i12],
        [r02 - sp.I * i02, r12 - sp.I * i12, d2]
    ])

def cb(z):
    Mref1 = config["Mref"]
    exponent = -2.3e12 * z / Mref1
    
    
    if hasattr(z, 'free_symbols') or isinstance(z, sp.Basic):
        return 1.0 - sp.exp(exponent)
    else:
        
        return 1.0 - np.exp(exponent)

def ch(z):
    Mref1 = config["Mref"]
    T_inv = z / Mref1
    
    if hasattr(z, 'free_symbols') or isinstance(z, sp.Basic):
        exp = lambda t: sp.exp(-t * T_inv)
    else:
        
        exp = lambda t: np.exp(-t * T_inv)
    return ((2/3 + 1/3 * exp(1e15)) - (2/3 - 14/23)*(1-exp(2e13)) -
            (14/23-2/5)*(1-exp(3e11)) - (2/5-4/13)*(1-exp(2e10)) -
            (4/13-3/10)*(1-exp(9e8)) - (3/10-1/4)*(1-exp(3e8)) -
            (1/4-2/11)*(1-exp(2e6)))


# Boltzmann Equations

def BoltzmannLepto(z, y_vec, contributions=None, params_sm=None):
    inv_glZl = 1.0 / (gl * Zetal)
    inv_gHZH = 1.0 / (gH * ZetaH)
    inv_gEZE = 1.0 / (gE * ZetaE)
    inv_Ynor = 1.0 / Ynor

    p = params_def if params_sm is None else params_sm
    
    if isinstance(y_vec[0], sp.Symbol): 
        YTDelta = build_hermitian_matrix_sym(y_vec[0:9])
        YDE = build_hermitian_matrix_sym(y_vec[9:18])
        y_newphys = y_vec[18:]
        eye_func = sp.eye
    else:
        def build_hermitian_matrix_num(v):
            return np.array([[v[0], v[3]+1j*v[4], v[5]+1j*v[6]],
                             [v[3]-1j*v[4], v[1], v[7]+1j*v[8]],
                             [v[5]-1j*v[6], v[7]-1j*v[8], v[2]]], dtype=np.complex128)
        YTDelta = build_hermitian_matrix_num(y_vec[0:9])
        YDE = build_hermitian_matrix_num(y_vec[9:18])
        y_newphys = y_vec[18:]
        eye_func = np.eye

    def get_param(param, z):
        return param(z) if callable(param) else param
    yE = get_param(p["yE"], z)

    def dagger(A):
        if isinstance(A, sp.Matrix):
            return A.H
        return A.T.conj()
        
    yET = dagger(yE)
    yETyE, yEyET = yET @ yE, yE @ yET

    trYDE, trYTDelta = YDE.trace(), YTDelta.trace()
    ydh_val = -ch(z) * (trYTDelta - 2*trYDE)
    YDH = ydh_val * eye_func(3)
    YDL = 2/15 * cb(z) * trYTDelta * eye_func(3) - YTDelta

    term_YDL_scaled = YDL * inv_glZl
    term_YDE_scaled = YDE * inv_gEZE

    if contributions and "HYPERCHARGES" in contributions:
        hypercharge_dict = contributions["HYPERCHARGES"]
        for idx, Y_val in hypercharge_dict.items():
            y_field_val = y_newphys[idx]
            contrib_field = y_field_val.trace() if hasattr(y_field_val, 'trace') else y_field_val
            YDH += ch(z) * (-2.0 * Y_val * contrib_field) * eye_func(n)

    gE_step = gammaE(z) * inv_Ynor
    rhs_newphys = []

    rhsLepto = {
        "YTDelta": ((gE_step * 0.5) * (yETyE @ term_YDL_scaled + term_YDL_scaled @ yETyE) - gE_step * (yETyE @ (YDH * inv_gHZH)) - gE_step * (yET @ term_YDE_scaled @ yE)),
        "YDE": (-(gE_step * 0.5) * (yEyET @ term_YDE_scaled + term_YDE_scaled @ yEyET) - gE_step * (yEyET @ (YDH * inv_gHZH)) + gE_step * (yE @ term_YDL_scaled @ yET))
    }
    if isinstance(y_vec[0], sp.Symbol):
        zero_matrix = sp.zeros(3, 3)
    else:
        zero_matrix = np.zeros((3, 3), dtype=np.complex128)
    
    YDQ = zero_matrix
    YDU = zero_matrix
    YDD = zero_matrix
    if contributions:
        for channel, new_functions in contributions.items():
            for func in new_functions:
                if channel == "NEW_DEGREES_OF_FREEDOM":
                    rhs_newphys.append(func(z, y_newphys, YDL, YDE, YDH, YDQ, YDU, YDD))
                elif channel in rhsLepto:
                    rhsLepto[channel] += func(z, y_newphys, YDL, YDE, YDH, YDQ, YDU, YDD)

    f_list = []
    for name in ["YTDelta", "YDE"]:
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


#  jac and RHS ode

def BE_RHSLE(y0_total, contributions=None, params_sm=None, background_funcs=None):
    import sys
    main_mod = sys.modules['__main__']
    if background_funcs is None:
        background_funcs = []

    total_vars = len(y0_total)

    z_sym = sp.symbols('z', real=True, positive=True)
    y_symbols = sp.symbols(f'y0:{total_vars}', real=True)

    p = params_def if params_sm is None else params_sm
    params_sym = p.copy()  
    orig_yE = None

    if callable(params_sym["yE"]):
        orig_yE = params_sym["yE"]
        
    dict_translate = {}

    sym_masks = {
    'gammaE': sp.Function("gammaE_sym"),
    'cb': sp.Function("cb_sym"),
    'ch': sp.Function("ch_sym")
}
    def yE_mock(z):
        return sp.Matrix([
            [sp.Function("yE11_sym")(z), sp.Function("yE12_sym")(z), sp.Function("yE13_sym")(z)],
            [sp.Function("yE21_sym")(z), sp.Function("yE22_sym")(z), sp.Function("yE23_sym")(z)],
            [sp.Function("yE31_sym")(z), sp.Function("yE32_sym")(z), sp.Function("yE33_sym")(z)],
        ])
    def gammaE_mock(z):
        return sym_masks['gammaE'](z)
    def cb_mock(z):
        return sym_masks['cb'](z)
    
    def ch_mock(z):
        return sym_masks['ch'](z)
    
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
    
    global gammaE, cb, ch

    orig_sm = gammaE
    orig_cb = cb
    orig_ch = ch
    
    gammaE = gammaE_mock
    cb = cb_mock
    ch = ch_mock
    if callable(params_sym["yE"]):
        params_sym["yE"] = yE_mock
    try:
        rhs_simbolic = BoltzmannLepto(z_sym, list(y_symbols), contributions, params_sym)
    finally:
        gammaE = orig_sm
        cb = orig_cb
        ch = orig_ch
        if orig_yE is not None:
            params_sym["yE"] = orig_yE
        for name, orig_func in orig_np.items():
            setattr(main_mod, name, orig_func)

# En lugar de hacer sympify a ciegas en el 'else':
    rhs_simbolic_pure = []
    for expr in rhs_simbolic:
        if hasattr(expr, 'as_explicit'):  
            rhs_simbolic_pure.append(expr.as_explicit())
        elif isinstance(expr, (sp.Basic, sp.Matrix)):
            rhs_simbolic_pure.append(expr)  # Si ya es SymPy, ¡no hagas nada!
        elif hasattr(expr, '__getitem__'):
            rhs_simbolic_pure.append(expr[0])
        else:
            rhs_simbolic_pure.append(sp.sympify(expr)) # Solo si es un int/float puro de Python
    
    # ============================================================
# Jacobian analítico
# ============================================================

    f_matrix = sp.Matrix(rhs_simbolic_pure)
    
    print("--- Analytical Jacobian Matrix ---")
    
    J_symbolic = f_matrix.jacobian(y_symbols)
    
    # --- Complexity of Jacobian entries ---
    
    
    # ============================================================
    # Traducción de funciones simbólicas -> funciones numéricas
    # ============================================================
    
    for name, orig_func in orig_np.items():
        dict_translate[f"{name}_sym"] = lambda z, i=None, f=orig_func: (
            f(z)[int(i)] if i is not None else f(z)
        )
    
    
    # yE(z), si es una función
    if orig_yE is not None:
        for i in range(3):
            for j in range(3):
                dict_translate[f"yE{i+1}{j+1}_sym"] = (
                    lambda z, ii=i, jj=j, f=orig_yE:
                    f(z)[ii, jj]
                )
    
    
    # Funciones mockeadas de fondo
    dict_translate.update({
        'gammaE_sym': lambda z: orig_sm(z),
        'cb_sym': lambda z: orig_cb(z),
        'ch_sym': lambda z: orig_ch(z)
    })
    
    
    # ============================================================
    # Modules para lambdify
    # ============================================================
    
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
    
    
    # ============================================================
    # Funciones numéricas finales
    # ============================================================
    
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
    
    
    print("End of Jacobian and Ode")
    
    return ode_analitic, jac_analitic, total_vars