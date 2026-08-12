import sympy as sp
import numpy as np
from scipy.special import kn
from ..config import config
import sympy as sp
import numpy as np
from scipy.special import kn
import time

# GLOBAL CONSTANTS AND PARAMETERS

gl, gH, gE, gQ, gU, gD = 2, 2, 1, 6, 3, 3
Zetal, ZetaH, ZetaE, ZetaQ, ZetaU, ZetaD = 1, 2, 1, 1, 1, 1
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

# CKM
theta12, theta23, theta13, delta = 0.227, 4.65e-2, 4.11e-3, 1.139j
V23 = np.array([[1, 0, 0], [0, np.cos(theta23), np.sin(theta23)], [0, -np.sin(theta23), np.cos(theta23)]], dtype=np.complex128)
V13 = np.array([[np.cos(theta13), 0, np.sin(theta13)*np.exp(-delta)], [0, 1, 0], [-np.sin(theta13)*np.exp(delta), 0, np.cos(theta13)]], dtype=np.complex128)
V12 = np.array([[np.cos(theta12), np.sin(theta12), 0], [-np.sin(theta12), np.cos(theta12), 0], [0, 0, 1]], dtype=np.complex128)
VCKM = V23 @ V13 @ V12

yU_def = np.array([[4.39e-6, 0, 0], [0, 1.98e-3, 0], [0, 0, 0.4454]], dtype=np.complex128)
yDdiag = np.array([[0.97e-5, 0, 0], [0, 1.72e-4, 0], [0, 0, 0.719e-2]], dtype=np.complex128)
yD_def = yDdiag @ np.conj(VCKM.T)
yE_def = np.array([[2.8e-6, 0, 0], [0, 5.9e-4, 0], [0, 0, 1e-2]], dtype=np.complex128)

params_def = {"yU": yU_def, "yD": yD_def, "yE": yE_def}

# =====================================================================
# Thermal functions
# =====================================================================

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

def gammaEW(z):
    Mref1 = config["Mref"]
    T = Mref1 / z
    return (GAMMA2_PRE * T**4) * _common_factor(z)
    
def gammaQCD(z):
    Mref1 = config["Mref"]
    T = Mref1 / z
    return (GAMMA3_PRE * T**4) * _common_factor(z)

def kv_safe(nu, z_val):
    val = kn(nu, z_val)
    return np.where(val == 0.0, np.finfo(np.float64).tiny, val)

# Construct a symbolic hermitian matrix
def build_hermitian_matrix_sym(vars_9):
    d0, d1, d2, r01, i01, r02, i02, r12, i12 = vars_9
    return sp.Matrix([
        [d0, r01 + sp.I * i01, r02 + sp.I * i02],
        [r01 - sp.I * i01, d1, r12 + sp.I * i12],
        [r02 - sp.I * i02, r12 - sp.I * i12, d2]
    ])


# Boltzman Equations (Standard Model + New Physics from the user)
def BoltzmannEQ(z, y_vec, contributions=None, params_sm=None):
    
    inv_glZl = 1.0 / (gl * Zetal)
    inv_gQZQ = 1.0 / (gQ * ZetaQ)
    inv_gUzu = 1.0 / (gU * ZetaU)
    inv_gDZz = 1.0 / (gD * ZetaD)
    inv_gHZH = 1.0 / (gH * ZetaH)
    inv_gEZE = 1.0 / (gE * ZetaE)
    inv_Ynor = 1.0 / Ynor

    p = params_def if params_sm is None else params_sm
    # Detection of Symbolic and Numerical
    if isinstance(y_vec[0], sp.Symbol): #First, for the analitical jacobian
        YDL = build_hermitian_matrix_sym(y_vec[0:9])
        YDE = build_hermitian_matrix_sym(y_vec[9:18])
        YDQ = build_hermitian_matrix_sym(y_vec[18:27])
        YDU = build_hermitian_matrix_sym(y_vec[27:36])
        YDD = build_hermitian_matrix_sym(y_vec[36:45])
        y_newphys = y_vec[45:]
        eye_func = sp.eye

    else:
        #For the Differential equation solver
        def build_hermitian_matrix_num(v):
            return np.array([[v[0], v[3]+1j*v[4], v[5]+1j*v[6]],
                             [v[3]-1j*v[4], v[1], v[7]+1j*v[8]],
                             [v[5]-1j*v[6], v[7]-1j*v[8], v[2]]], dtype=np.complex128)
        YDL = build_hermitian_matrix_num(y_vec[0:9])
        YDE = build_hermitian_matrix_num(y_vec[9:18])
        YDQ = build_hermitian_matrix_num(y_vec[18:27])
        YDU = build_hermitian_matrix_num(y_vec[27:36])
        YDD = build_hermitian_matrix_num(y_vec[36:45])
        y_newphys = y_vec[45:]
        eye_func = np.eye
        
       
    def get_param(param, z):
        return param(z) if callable(param) else param
    yU = get_param(p["yU"], z)
    yD = get_param(p["yD"], z)
    yE = get_param(p["yE"], z)

    def dagger(A):
        if isinstance(A, sp.Matrix):
            return A.H
        return A.T.conj()
        
    yUT, yDT, yET = dagger(yU), dagger(yD), dagger(yE)
    
  
    yETyE, yEyET = yET @ yE, yE @ yET
    yUTyU, yDTyD = yUT @ yU, yDT @ yD

    term_YDQ_scaled = YDQ * inv_gQZQ
    term_YDL_scaled = YDL * inv_glZl
    term_YDE_scaled = YDE * inv_gEZE

    trYDL, trYDQ, trYDU, trYDD, trYDE = YDL.trace(), YDQ.trace(), YDU.trace(), YDD.trace(), YDE.trace()

    cew_val = (gammaEW(z) * 0.25 * inv_Ynor) * (trYDL * inv_glZl + 3.0 * trYDQ * inv_gQZQ)
    Cew = cew_val * eye_func(n)

    cqcd_val = (gammaQCD(z) / 6.0 * inv_Ynor) * (2.0 * trYDQ * inv_gQZQ - trYDU * inv_gUzu - trYDD * inv_gDZz)
    CQCD = cqcd_val * eye_func(n)

    ydh_val = (1.0/3.0) * (-trYDQ - 4.0 * trYDU + 2.0 * trYDD + 3.0 * trYDL + 6.0 * trYDE)
    YDH = ydh_val * eye_func(n)
    if contributions and "HYPERCHARGES" in contributions:
        hypercharge_dict = contributions["HYPERCHARGES"]
        for idx, Y_val in hypercharge_dict.items():
            
            y_field_val = y_newphys[idx]
            
            if hasattr(y_field_val, 'trace'):
                contrib_field = y_field_val.trace()
            else:
                contrib_field = y_field_val
            
            YDH += (-2.0 * Y_val * contrib_field) * eye_func(n)
            
    gE_step = gammaE(z) * inv_Ynor
    gU_step = gammaU(z) * inv_Ynor
    gD_step = gammaD(z) * inv_Ynor

    rhs_newphys = []

    rhsSM = {
        "YDL": (-Cew - (gE_step * 0.5) * (yETyE @ term_YDL_scaled + term_YDL_scaled @ yETyE) + gE_step * (yETyE @ (YDH * inv_gHZH)) + gE_step * (yET @ term_YDE_scaled @ yE)),
        "YDE": (-(gE_step * 0.5) * (yEyET @ term_YDE_scaled + term_YDE_scaled @ yEyET) - gE_step * (yEyET @ (YDH * inv_gHZH)) + gE_step * (yE @ term_YDL_scaled @ yET)),
        "YDQ": (-3.0*Cew - 2.0*CQCD - (gU_step * 0.5) * (yUTyU @ term_YDQ_scaled + term_YDQ_scaled @ yUTyU) - gU_step * (yUTyU @ (YDH * inv_gHZH)) + gU_step * (yUT @ (YDU * inv_gUzu) @ yU) - (gD_step * 0.5) * (yDTyD @ term_YDQ_scaled + term_YDQ_scaled @ yDTyD) + gD_step * (yDTyD @ (YDH * inv_gHZH)) + gD_step * (yDT @ (YDD * inv_gDZz) @ yD)),
        "YDU": (CQCD - (gU_step * 0.5) * ((yU@yUT) @ (YDU * inv_gUzu) + (YDU * inv_gUzu) @ (yU@yUT)) + gU_step * (yU @ yUT @ (YDH * inv_gHZH)) + gU_step * (yU @ term_YDQ_scaled @ yUT)),
        "YDD": (CQCD - (gD_step * 0.5) * ((yD@yDT) @ (YDD * inv_gDZz) + (YDD * inv_gDZz) @ (yD@yDT)) -gD_step * (yD @ yDT @ (YDH * inv_gHZH)) + gD_step * (yD @ term_YDQ_scaled @ yDT))
    }
 #The new degrees of freedom and the Washout and Source terms
    if contributions:
        for channel, new_functions in contributions.items():
            for func in new_functions:
                if channel == "NEW_DEGREES_OF_FREEDOM":
                    rhs_newphys.append(func(z,y_newphys, YDL, YDE, YDH, YDQ, YDU, YDD))
                elif channel in rhsSM:
                    rhsSM[channel] += func(z,y_newphys, YDL, YDE, YDH, YDQ, YDU, YDD)

    # Flattening for the solver
    f_list = []
    for name in ["YDL", "YDE", "YDQ", "YDU", "YDD"]:
        dm = rhsSM[name]
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
        # For the new degrees of freedom
        for item in rhs_newphys:
            if hasattr(item, '__iter__') or isinstance(item, sp.Matrix): #If it is a vector or matrix, flatten
                f_list.extend(list(item))
            else: #if is a scalar, add without problem
                f_list.append(item)

    return f_list if isinstance(y_vec[0], sp.Symbol) else np.array(f_list)


# =====================================================================
# Jacobian and ode calculation
# =====================================================================
def BE_RHS(y0_total, contributions=None, params_sm=None, background_funcs=None):
    import sys
    main_mod = sys.modules['__main__']
    if background_funcs is None:
        background_funcs = []

    total_vars = len(y0_total)

    z_sym = sp.symbols('z', real=True, positive=True)
    y_symbols = sp.symbols(f'y0:{total_vars}', real=True)
    y_matrix_sym = sp.Matrix(y_symbols)

    p = params_def if params_sm is None else params_sm
    params_sym = p.copy()
    orig_yE = None

    if callable(params_sym["yE"]):
        orig_yE = params_sym["yE"]
        
    dict_translate = {}

    sm_funcs = ['gammaE', 'gammaU', 'gammaD', 'gammaEW', 'gammaQCD']

    sym_masks = {}
    for name in sm_funcs:
        sym_masks[name] = sp.Function(f"{name}_sym")
    def yE_mock(z):
        return sp.Matrix([
            [sp.Function("yE11_sym")(z), sp.Function("yE12_sym")(z), sp.Function("yE13_sym")(z)],
            [sp.Function("yE21_sym")(z), sp.Function("yE22_sym")(z), sp.Function("yE23_sym")(z)],
            [sp.Function("yE31_sym")(z), sp.Function("yE32_sym")(z), sp.Function("yE33_sym")(z)],
        ])
    def gammaEW_mock(z):
        return sym_masks['gammaEW'](z)
    
    def gammaQCD_mock(z):
        return sym_masks['gammaQCD'](z)
    
    def gammaE_mock(z):
        return sym_masks['gammaE'](z)
    
    def gammaU_mock(z):
        return sym_masks['gammaU'](z)
    
    def gammaD_mock(z):
        return sym_masks['gammaD'](z)
    
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
    
    global gammaEW, gammaQCD, gammaE, gammaU, gammaD
    
    orig_sm = (
        gammaEW,
        gammaQCD,
        gammaE,
        gammaU,
        gammaD
    )
    
    gammaEW = gammaEW_mock
    gammaQCD = gammaQCD_mock
    gammaE = gammaE_mock
    gammaU = gammaU_mock
    gammaD = gammaD_mock
    if callable(params_sym["yE"]):
        params_sym["yE"] = yE_mock
        
    try:
        rhs_simbolic = BoltzmannEQ(z_sym, list(y_symbols), contributions, params_sym)
    finally:
        gammaEW, gammaQCD, gammaE, gammaU, gammaD = orig_sm
        if orig_yE is not None:
            params_sym["yE"] = orig_yE
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
    print("--- Jacobian Analitic Matrix---")
    start_time = time.time()
    J_symbolic = f_matrix.jacobian(y_symbols)
    
    # --- Complexity of Jacobian entries --
    
    

    for name, orig_func in orig_np.items():
        dict_translate[f"{name}_sym"] = lambda z, i=None, f=orig_func: (
            f(z)[int(i)] if i is not None else f(z)
        )
    if orig_yE is not None:

        for i in range(3):
            for j in range(3):
                dict_translate[f"yE{i+1}{j+1}_sym"] = (
                    lambda z, ii=i, jj=j, f=orig_yE: f(z)[ii, jj]
                )
       
    dict_translate.update({
    'gammaE_sym': lambda z: orig_sm[2](z),
    'gammaU_sym': lambda z: orig_sm[3](z),
    'gammaD_sym': lambda z: orig_sm[4](z),
    'gammaEW_sym': lambda z: orig_sm[0](z),
    'gammaQCD_sym': lambda z: orig_sm[1](z)
})
    map_modules = [{**dict_translate}, 'numpy']
    
    print("---Lambyfy ---")
        
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
    print(f"Time of Jacobian calcultaion = {time.time() - start_time:.4f} seconds.")
    print("End of Jacobian and Ode")
    return ode_analitic, jac_analitic, total_vars