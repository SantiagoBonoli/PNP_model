# -*- coding: utf-8 -*-
"""
Nanoporo PET en KCl 0.1 M: modelo PNP 2D axisimetrico (r,z)
con reservorios especulares de 10 um y nanoporo de z=0 a z=L.

Convenciones:
  - phi = 0 en el extremo del reservorio IZQUIERDO (z = z[0]).
  - phi = V en el extremo del reservorio DERECHO  (z = z[-1]).
  - I > 0 para V > 0 (SIGNO_I = -1; poner +1 para "corriente en sentido +z").

Novedades de esta version
  1. Malla axial del poro por equidistribucion (funcion monitor): celdas finas
     en la punta (Rtip) y en las bocas, gruesas en el cuerpo del poro.
  2. Celda radial de pared fina (dw0) solo dentro del poro; en accesos y
     reservorios crece como sqrt(R/Rporo).
  3. GPU opcional y apagada por defecto (usar_gpu(True/False)); el Jacobiano
     transfiere solo las entradas necesarias.
  4. MODOS DE CARGA (parametro modo_carga de Malla):
       "superficial"   : solo carga superficial sigma en la pared del poro.
       "vol_z"         : carga VOLUMETRICA rho_vol (C/m3) en el rango axial
                         z_vol = (za, zb) nm, llenando toda la seccion (de la
                         pared al centro). Fuera de ese rango, carga superficial.
       "vol_radial"    : carga VOLUMETRICA rho_vol en una capa de espesor S_vol (nm)
                         medida desde la pared hacia el centro, en el rango axial
                         z_vol (por defecto todo el poro, (0, L)). Fuera de z_vol,
                         carga superficial.
       "vol_combinado" : zona LLENA rho_vol en z_vol = (za, zb) (toda la seccion)
                         + capa radial de espesor S_vol con densidad rho_rad
                         (por defecto = rho_vol) en z_rad = (ra, rb); si z_rad es
                         None, la capa radial ocupa el RESTO del poro (los tramos
                         de [0, L] fuera de z_vol). z_rad no puede solaparse con
                         z_vol. Fuera de ambas zonas, carga superficial.
     En las zonas volumetricas la carga superficial se APAGA (la carga volumetrica
     reemplaza a la pared cargada); con sigma_en_zona=True se mantiene tambien la
     de pared. La carga volumetrica entra en Poisson como
         lap(phi) = -(c+ - c- + X)/ell^2,   X = rho/(F*C0)
     con factores de volumen por solapamiento exacto entre cada volumen de
     control y la region cargada; los bordes axiales de las zonas caen sobre nodos
     y, si una zona es corta, se malla con celdas uniformes de ~dz_vol nm.

Ejemplos:
    usar_gpu(False)
    m = Malla(20, 295, 6.0, L=12000.0, modo_carga="superficial")
    m = Malla(20, 295, 6.0, L=12000.0, modo_carga="vol_z",
              rho_vol=-8e6, z_vol=(0.0, 50.0))
    m = Malla(20, 295, 6.0, L=12000.0, modo_carga="vol_radial",
              rho_vol=-8e6, S_vol=10.0)
    # punta llena 0-50 nm (rho_vol) + capa radial de 10 nm (rho_rad) en el resto
    m = Malla(20, 295, 6.0, L=12000.0, modo_carga="vol_combinado",
              rho_vol=-8e6, z_vol=(0.0, 50.0), S_vol=10.0, rho_rad=-2e6)
    m.resumen_carga(sigma)
    I, sol = curva_IV(m, np.linspace(-1, 1, 21), sigma, usar_refinamiento=True)
"""

import traceback
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import brentq

# ----------------------------- GPU opcional -----------------------------
HAS_GPU = False
USAR_GPU = False          # con < ~2e5 incognitas, CPU suele ser igual o mas rapido
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()      # falla si no hay GPU/driver utilizable
    HAS_GPU = True
except Exception:
    import numpy as cp                    # fallback: cp = numpy


def usar_gpu(flag):
    """Llamar ANTES de construir cualquier Malla (los arrays quedan en el
    dispositivo activo)."""
    global USAR_GPU
    USAR_GPU = bool(flag) and HAS_GPU
    print(f"--- Residuo en {'GPU (CuPy)' if USAR_GPU else 'CPU (NumPy)'} ---")


def _xp():
    return cp if USAR_GPU else np


def _to_cpu(a):
    return cp.asnumpy(a) if USAR_GPU else np.asarray(a)


def _to_dev(a):
    return cp.asarray(a) if USAR_GPU else np.asarray(a)


# ----------------------------- constantes -----------------------------
F = 96485.33
R_GAS = 8.314462
T = 298.15
N_A = 6.02214076e23
E_CHARGE = 1.602176634e-19
VT = R_GAS * T / F
EPS = 78.5 * 8.8541878128e-12          # F/m, permitividad del agua
C0 = 100.0                             # mol/m3 = 0.1 M
D_K, D_CL = 1.957e-9, 2.032e-9
FACT_D = 0.86                          # ajusta kappa(0.1 M KCl) ~= 1.29 S/m
D0 = D_K * FACT_D
D_P = 1.0
D_M = D_CL / D_K
ELL2 = EPS * R_GAS * T / (F**2 * C0) * 1e18  # nm2; lambda_D^2 = ELL2/2
SIGHAT = F / (EPS * R_GAS * T) * 1e-9       # (C/m2) -> (1/nm)

SIGNO_I = -1.0


def _B2(x):
    """Bernoulli estable para Scharfetter-Gummel: devuelve (B(x), B(-x))."""
    xp = _xp()
    y = xp.minimum(xp.abs(x), 700.0)
    small = y < 1e-8
    ys = xp.where(small, 1.0, y)
    Bp = xp.where(small, 1.0 - 0.5 * y, ys / xp.expm1(ys))   # B(+|x|)
    Bn = Bp + y                                               # B(-|x|)
    pos = x >= 0.0
    return xp.where(pos, Bp, Bn), xp.where(pos, Bn, Bp)


# ----------------------------- nodos 1D -----------------------------
def _nodos_segmento(x0, x1, n, dx0):
    """Nodos 1D con agrupamiento tanh en ambos extremos del segmento."""
    if n < 2:
        raise ValueError("Cada segmento axial necesita al menos dos celdas.")
    largo = float(x1 - x0)
    s = np.linspace(0.0, 1.0, n + 1)
    if dx0 <= 0.0 or dx0 >= largo / n:
        return np.linspace(x0, x1, n + 1)

    def zf(beta):
        return x0 + largo * 0.5 * (
            1.0 + np.tanh(beta * (s - 0.5)) / np.tanh(beta / 2.0)
        )

    beta = brentq(lambda b: zf(b)[1] - (x0 + dx0), 1e-3, 80.0)
    z = zf(beta)
    z[0], z[-1] = x0, x1
    return z


def _perfil_poro(xf, Rtip, Rbase, L_h, L):
    if abs(L_h) < 1e-8:
        g = (xf / L)
    else:
        g = (1-np.exp(-(L_h * xf/L)**0.8)) / (1-np.exp(-L_h**0.8))
    return Rtip + (Rbase - Rtip) * g


def _nodos_poro(Rtip, Rbase, L_h, L, n, A=1.0, B=15.0, bordes=(), uniformes=(),
                dz_unif=2.0, p=1.0, ell_fac=3.0):
    """
    Nodos del poro (n celdas en total). Funcion monitor
        M(z) = 1 + A*(Rmax/rho)**p + B*[exp(-z/ell) + exp(-(L-z)/ell) + sum_e exp(-|z-z_e|/ell)]
    (celdas finas en la punta, en las bocas y en los bordes z_e de la zona cargada).
    El poro se parte en los 'bordes' (nodos exactos en esos z). Los tramos listados
    en 'uniformes' (a,b) se malla con celdas uniformes de ~dz_unif nm si son cortos;
    el resto de las celdas se reparte entre los demas tramos segun la masa de M.
    """
    xf = np.linspace(0.0, L, 16001)
    rho = _perfil_poro(xf, Rtip, Rbase, L_h, L)
    ell = ell_fac * min(Rtip, Rbase)
    M = (1.0 + A * (max(Rtip, Rbase) / rho) ** p
         + B * (np.exp(-xf / ell) + np.exp(-(L - xf) / ell)))
    for ze in bordes:
        M = M + B * np.exp(-np.abs(xf - ze) / ell)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (M[1:] + M[:-1]) * np.diff(xf))])

    bp = sorted(set([0.0, float(L)] + [float(b) for b in bordes if 0.0 < b < L]))
    segs = list(zip(bp[:-1], bp[1:]))

    def es_unif(a, b):
        if not any(abs(a - ua) < 1e-9 and abs(b - ub) < 1e-9 for ua, ub in uniformes):
            return False
        return int(np.ceil((b - a) / dz_unif)) <= n // 2

    n_i = [None] * len(segs)
    for k, (a, b) in enumerate(segs):
        if es_unif(a, b):
            n_i[k] = max(4, int(np.ceil((b - a) / dz_unif)))
    mon = [k for k in range(len(segs)) if n_i[k] is None]
    rest = n - sum(v for v in n_i if v is not None)
    if not mon or rest < 8 * len(mon):
        raise ValueError("Nz_poro insuficiente para la zona cargada: subir Nz_poro "
                         "o dz_vol.")
    mass = np.array([np.interp(segs[k][1], xf, cum) - np.interp(segs[k][0], xf, cum)
                     for k in mon])
    cells = np.maximum(6, np.rint(rest * mass / mass.sum()).astype(int))
    while cells.sum() > rest:
        cells[np.argmax(cells)] -= 1
    while cells.sum() < rest:
        cells[np.argmax(mass / cells)] += 1
    for k, c in zip(mon, cells):
        n_i[k] = int(c)

    z = [0.0]
    for k, (a, b) in enumerate(segs):
        if k in mon:
            ca, cb = np.interp(a, xf, cum), np.interp(b, xf, cum)
            seg = np.interp(np.linspace(ca, cb, n_i[k] + 1), cum, xf)
            seg[0], seg[-1] = a, b
        else:
            seg = np.linspace(a, b, n_i[k] + 1)
        z.extend(seg[1:])
    z = np.array(z)
    z[-1] = L
    return z


def _forzar_nodo(z, zt):
    """Mueve el nodo interior mas cercano a zt para que z contenga zt exacto."""
    k = int(np.argmin(np.abs(z - zt)))
    if k == 0 or k == len(z) - 1:
        return z
    z = z.copy()
    z[k] = zt
    if not np.all(np.diff(z) > 0):
        raise ValueError("No se pudo insertar el nodo z de borde; aumentar Nz_poro.")
    return z


# ----------------------------- malla -----------------------------
class Malla:
    """Malla axisimetrica con reservorios cuadrados y nanoporo de z=0 a z=L."""

    def __init__(
        self,
        Rtip,
        Rbase,
        L_h,
        L=12000.0,
        Rres=10000.0,
        Laccess=500.0,
        Nz_poro=120,
        Nz_res=50,
        Nz_access=40,
        Neta=24,
        dz0=0.05,           # celda en extremos del poro (solo malla_poro="tanh") (nm)
        dw0=0.05,           # celda radial en la pared dentro del poro (nm)
        dz0_access=0.5,
        dz0_res=20.0,
        Laccess_R=None,
        Nz_access_R=None,
        dz0_access_R=None,
        malla_poro="monitor",   # "monitor" | "tanh"
        A_mon=1.0,
        B_mon=15.0,
        # ---- carga del poro ----
        modo_carga="superficial",   # "superficial" | "vol_z" | "vol_radial" | "vol_combinado"
        rho_vol=0.0,                # C/m3: zona llena (vol_z, vol_combinado) o capa (vol_radial)
        z_vol=None,                 # (za, zb) nm
        S_vol=None,                 # espesor radial desde la pared (nm)
        rho_rad=None,               # C/m3 de la capa radial en "vol_combinado" (None -> rho_vol)
        z_rad=None,                 # (ra, rb) nm de la capa radial en "vol_combinado" (None -> resto)
        sigma_en_zona=False,        # True: mantener tambien la carga de pared en las zonas
        dz_vol=2.0,                 # celda axial objetivo (nm) en zonas volumetricas cortas
    ):
        if min(Rtip, Rbase, L) <= 0:
            raise ValueError("Rtip, Rbase y L deben ser positivos.")
        if Rbase < Rtip:
            warnings.warn("Rbase < Rtip: el poro se estrecha en la direccion +z.")
        if modo_carga not in ("superficial", "vol_z", "vol_radial", "vol_combinado"):
            raise ValueError("modo_carga debe ser 'superficial', 'vol_z', 'vol_radial' "
                             "o 'vol_combinado'.")
        if modo_carga in ("vol_z", "vol_combinado") and z_vol is None:
            raise ValueError(f"modo '{modo_carga}' requiere z_vol=(za, zb).")
        if modo_carga in ("vol_radial", "vol_combinado") and (S_vol is None or S_vol <= 0):
            raise ValueError(f"modo '{modo_carga}' requiere S_vol > 0 (nm).")
        if modo_carga == "vol_radial" and z_vol is None:
            z_vol = (0.0, L)

        def _tramo(zz, nombre):
            a, b = max(float(zz[0]), 0.0), min(float(zz[1]), float(L))
            if not b > a:
                raise ValueError(f"{nombre} debe cumplir 0 <= a < b <= L.")
            return (a, b)

        if z_vol is not None:
            z_vol = _tramo(z_vol, "z_vol")
        if z_rad is not None:
            if modo_carga != "vol_combinado":
                raise ValueError("z_rad solo se usa con modo 'vol_combinado'.")
            z_rad = _tramo(z_rad, "z_rad")
            if min(z_vol[1], z_rad[1]) - max(z_vol[0], z_rad[0]) > 1e-9:
                raise ValueError("z_rad no puede solaparse con z_vol.")

        self.Rtip = float(Rtip)
        self.Rbase = float(Rbase)
        self.L_h = float(L_h)
        self.L = float(L)

        self.Rres = float(Rres)
        self.Lres = self.Rres
        self.Laccess = float(Laccess)

        self.N = int(Neta)
        self.Nz_poro = int(Nz_poro)
        self.Nz_res = int(Nz_res)
        self.Nz_access = int(Nz_access)

        self.dz0 = float(dz0)
        self.dw0 = float(dw0)
        self.dz0_access = float(dz0_access)
        self.dz0_res = float(dz0_res)
        self.malla_poro = malla_poro
        self.A_mon = float(A_mon)
        self.B_mon = float(B_mon)

        self.modo_carga = modo_carga
        self.rho_vol = float(rho_vol)
        self.rho_rad = self.rho_vol if rho_rad is None else float(rho_rad)
        self.z_vol = z_vol
        self.z_rad = z_rad
        self.S_vol = None if S_vol is None else float(S_vol)
        self.sigma_en_zona = bool(sigma_en_zona)
        self.dz_vol = float(dz_vol)

        # Zonas volumetricas: lista de (z0, z1, rho C/m3, S nm | None = seccion completa)
        zonas = []
        if modo_carga == "vol_z":
            zonas.append((z_vol[0], z_vol[1], self.rho_vol, None))
        elif modo_carga == "vol_radial":
            zonas.append((z_vol[0], z_vol[1], self.rho_vol, self.S_vol))
        elif modo_carga == "vol_combinado":
            za, zb = z_vol
            zonas.append((za, zb, self.rho_vol, None))
            if z_rad is None:
                if za > 0.0:
                    zonas.append((0.0, za, self.rho_rad, self.S_vol))
                if zb < float(L):
                    zonas.append((zb, float(L), self.rho_rad, self.S_vol))
            else:
                zonas.append((z_rad[0], z_rad[1], self.rho_rad, self.S_vol))
        self.zonas = [zn for zn in zonas if zn[2] != 0.0]
        self.hay_vol = len(self.zonas) > 0
        self.f_fijo = 1.0          # factor de continuacion de la carga fija

        if self.Rres <= max(self.Rtip, self.Rbase):
            raise ValueError("Rres debe ser mayor que Rtip y Rbase.")
        if self.Laccess <= 0:
            raise ValueError("Laccess debe ser positivo.")

        # Acceso derecho: si se pasa Laccess_R se respeta tal cual (p. ej. igual al
        # izquierdo -> malla especular); si es None se escala con la razon de
        # (Rres - R) para conservar la misma pendiente de la boca en ambos lados.
        if Laccess_R is None:
            self.Laccess_R = self.Laccess * (
                abs(self.Rres - self.Rbase) /
                abs(self.Rres - self.Rtip)
            )
        else:
            self.Laccess_R = float(Laccess_R)
        self.Nz_access_R = int(self.Nz_access if Nz_access_R is None else Nz_access_R)
        self.dz0_access_R = float(self.dz0_access if dz0_access_R is None else dz0_access_R)
        if self.Laccess_R <= 0:
            raise ValueError("Laccess_R debe ser positivo.")
        self.simetrica = (
            abs(self.Laccess_R - self.Laccess) < 1e-12
            and self.Nz_access_R == self.Nz_access
            and abs(self.dz0_access_R - self.dz0_access) < 1e-12
        )

        self.z_resL0 = -(self.Lres + self.Laccess)
        self.z_resL1 = -self.Laccess
        self.z_poro0 = 0.0
        self.z_poro1 = self.L
        self.z_accR1 = self.L + self.Laccess_R
        self.z_total = self.z_accR1 + self.Lres

        z_resL = _nodos_segmento(self.z_resL0, self.z_resL1, self.Nz_res, self.dz0_res)
        z_accL = _nodos_segmento(self.z_resL1, self.z_poro0, self.Nz_access, self.dz0_access)

        # --- nodos del poro ---
        bordes, unif = [], []
        for (a, b, _, _) in self.zonas:
            bordes += [x for x in (a, b) if 0.0 < x < self.L]
            unif.append((a, b))
        bordes = sorted(set(bordes))
        if malla_poro == "monitor":
            z_poro = self.z_poro0 + _nodos_poro(
                self.Rtip, self.Rbase, self.L_h, self.L, self.Nz_poro,
                A=self.A_mon, B=self.B_mon, bordes=bordes, uniformes=unif,
                dz_unif=self.dz_vol)
        else:
            z_poro = _nodos_segmento(self.z_poro0, self.z_poro1, self.Nz_poro, self.dz0)
            for b in bordes:
                z_poro = _forzar_nodo(z_poro, self.z_poro0 + b)

        if self.simetrica:
            z_accR = self.L - z_accL[::-1]
            z_resR = self.L - z_resL[::-1]
        else:
            z_accR = _nodos_segmento(self.z_poro1, self.z_accR1, self.Nz_access_R,
                                     self.dz0_access_R)
            z_resR = _nodos_segmento(self.z_accR1, self.z_total, self.Nz_res,
                                     self.dz0_res)

        z_parts = [z_resL, z_accL[1:], z_poro[1:], z_accR[1:], z_resR[1:]]
        self.z = np.concatenate(z_parts)
        self.Nz = len(self.z) - 1

        z = self.z
        rho = np.empty_like(z)
        m_resL = z <= self.z_resL1
        m_accL = (z > self.z_resL1) & (z <= self.z_poro0)
        m_poro = (z > self.z_poro0) & (z <= self.z_poro1)
        m_accR = (z > self.z_poro1) & (z <= self.z_accR1)
        m_resR = z > self.z_accR1

        rho[m_resL] = self.Rres
        rho[m_resR] = self.Rres

        s = (z[m_accL] - self.z_resL1) / self.Laccess
        smooth = s**2 * (3.0 - 2.0 * s)
        rho[m_accL] = self.Rres + (self.Rtip - self.Rres) * smooth

        rho[m_poro] = _perfil_poro(z[m_poro] - self.z_poro0,
                                   self.Rtip, self.Rbase, self.L_h, self.L)

        s = (z[m_accR] - self.z_poro1) / self.Laccess_R
        smooth = s**2 * (3.0 - 2.0 * s)
        rho[m_accR] = self.Rbase + (self.Rres - self.Rbase) * smooth

        rho[np.argmin(np.abs(z - self.z_resL1))] = self.Rres
        rho[np.argmin(np.abs(z - self.z_poro0))] = self.Rtip
        rho[np.argmin(np.abs(z - self.z_poro1))] = self.Rbase
        rho[np.argmin(np.abs(z - self.z_accR1))] = self.Rres
        self.rho = rho

        self.pared_cargada = (z >= self.z_poro0 - 1e-10) & (z <= self.z_poro1 + 1e-10)

        # --- malla radial: celda de pared fina solo donde hay carga (poro) ---
        Rmax_poro = max(self.Rtip, self.Rbase)
        r = np.zeros((self.Nz + 1, self.N + 1))
        k = np.arange(self.N + 1)
        for i in range(self.Nz + 1):
            Ri = rho[i]
            dw0_i = dw0 * max(1.0, Ri / Rmax_poro) ** 0.5
            if Ri <= self.N * dw0_i:
                d = Ri * k / self.N
            else:
                def fq(q, dw0_i=dw0_i, Ri=Ri):
                    return dw0_i * (q**self.N - 1.0) / (q - 1.0) - Ri
                q = brentq(fq, 1.0 + 1e-9, 10.0)
                d = dw0_i * (q**k - 1.0) / (q - 1.0)
            d[-1] = Ri
            r[i, :] = (Ri - d)[::-1]
        r[:, 0] = 0.0
        self.r = r

        # ----------------- precomputo en GPU/CPU -----------------
        xp = _xp()
        r_xp = xp.asarray(r)

        self.z_gpu = xp.asarray(self.z)
        self.rz_gpu = xp.asarray(np.gradient(r, z, axis=0, edge_order=2))
        self.rxi_gpu = xp.asarray(np.gradient(r, axis=1, edge_order=1))
        self.rf_gpu = 0.5 * (r_xp[:, 1:] + r_xp[:, :-1])

        rxif = r_xp[:, 1:] - r_xp[:, :-1]
        self.rzf_gpu = 0.5 * (self.rz_gpu[:, 1:] + self.rz_gpu[:, :-1])
        self.axi_gpu = (1.0 + self.rzf_gpu**2) / rxif

        hz_np = np.diff(z)[:, None]
        self.hz_gpu = xp.asarray(hz_np)
        self.ihz_gpu = xp.asarray(1.0 / hz_np)

        rzz = (r_xp[1:] - r_xp[:-1]) / self.hz_gpu
        rxz = 0.5 * (self.rxi_gpu[1:] + self.rxi_gpu[:-1])
        self.kz_gpu = rzz / rxz

        rbar = 0.5 * (r_xp[1:] + r_xp[:-1])
        self.Aface_gpu = self._areas(rbar, xp=xp)

        dzc = np.zeros(self.Nz + 1)
        dzc[1:-1] = 0.5 * (z[2:] - z[:-2])
        self.dzc_gpu = xp.asarray(dzc)[:, None]

        vol_val = self._areas(r_xp, xp=xp) * self.dzc_gpu
        vol_val[0, :] = 1.0
        vol_val[-1, :] = 1.0
        self.vol_gpu = xp.asarray(vol_val)

        # Peso de carga de pared por nodo: fraccion de la celda de control dentro del
        # poro [0, L] y FUERA de las zonas volumetricas (que reemplazan a la pared)
        w = self._frac_axial(self.z_poro0, self.z_poro1)
        if self.hay_vol and not self.sigma_en_zona:
            for (a, b, _, _) in self.zonas:
                w = w - self._frac_axial(a, b)
            w = np.clip(w, 0.0, None)
        self.pared_peso = w
        rho_xp = xp.asarray(rho)
        self.sqrt_pared_gpu = (
            rho_xp * xp.sqrt(1.0 + self.rz_gpu[:, -1]**2) * xp.asarray(w)
        )

        # ----------------- carga volumetrica fija (capa desde el tip) -----------------
        self.Xfix_gpu = xp.asarray(self._carga_fija())

        # Pesos de la derivada nodal en z (3 puntos, malla no uniforme)
        hs = np.diff(z)
        hm, hp = hs[:-1], hs[1:]
        self.dz_wm = xp.asarray((-hp / (hm * (hm + hp)))[:, None])
        self.dz_w0 = xp.asarray(((hp - hm) / (hm * hp))[:, None])
        self.dz_wp = xp.asarray((hm / (hp * (hm + hp)))[:, None])
        self.dz_ih0 = float(1.0 / hs[0])
        self.dz_ih1 = float(1.0 / hs[-1])

        self._patron = None
        self._lu = None

    # ---------- helpers de carga ----------
    def _frac_axial(self, a, b):
        """Fraccion axial de la celda de control de cada nodo que cae en [a, b]."""
        z = self.z
        zl = np.empty_like(z)
        zr = np.empty_like(z)
        zl[0], zl[1:] = z[0], 0.5 * (z[1:] + z[:-1])
        zr[-1], zr[:-1] = z[-1], 0.5 * (z[1:] + z[:-1])
        ov = np.clip(np.minimum(zr, b) - np.maximum(zl, a), 0.0, None)
        den = zr - zl
        return np.where(den > 0, ov / np.where(den > 0, den, 1.0), 0.0)

    def _frac_radial(self, S):
        """Fraccion del anillo nodal (i,j) dentro de rho(z)-S <= r <= rho(z).
        S=None -> 1 (seccion completa)."""
        r, rho = self.r, self.rho
        if S is None:
            return np.ones_like(r)
        rh = 0.5 * (r[:, 1:] + r[:, :-1])
        up = np.concatenate([rh, r[:, -1:]], axis=1)
        lo = np.concatenate([np.zeros((r.shape[0], 1)), rh], axis=1)
        rmin = (rho - S)[:, None]
        lo_c = np.maximum(lo, rmin)
        up_c = np.maximum(up, lo_c)
        return (up_c**2 - lo_c**2) / np.maximum(up**2 - lo**2, 1e-300)

    def _X_zona(self, a, b, rho_c, S):
        """X/C0 de una zona: fraccion cargada del volumen de control * rho/(F*C0)."""
        return (rho_c / (F * C0)) * self._frac_axial(a, b)[:, None] * self._frac_radial(S)

    def _carga_fija(self):
        """X(i,j) adimensional total = suma de las zonas volumetricas."""
        X = np.zeros((self.Nz + 1, self.N + 1))
        for (a, b, rho_c, S) in self.zonas:
            X += self._X_zona(a, b, rho_c, S)
        return X

    def resumen_carga(self, sigma=None):
        """Imprime la carga de cada zona volumetrica y (si se pasa sigma en C/m2) la
        superficial, en cargas elementales."""
        print(f"Modo de carga: {self.modo_carga}")
        vol = _to_cpu(self.vol_gpu)
        tot = 0.0
        for (a, b, rho_c, S) in self.zonas:
            Ne = np.sum(self._X_zona(a, b, rho_c, S) * C0 * vol) * 2.0 * np.pi * 1e-27 * N_A
            tot += Ne
            rad = "seccion completa" if S is None else f"capa de {S:g} nm desde la pared"
            print(f"  volumetrica z en [{a:g}, {b:g}] nm, {rad}: rho = {rho_c:.3e} C/m3 "
                  f"(X = {rho_c / F:.3g} mol/m3): {Ne:+.1f} e")
        if len(self.zonas) > 1:
            print(f"  volumetrica total: {tot:+.1f} e")
        if sigma is not None:
            area = float(np.sum(_to_cpu(self.sqrt_pared_gpu) * _to_cpu(self.dzc_gpu)[:, 0])) \
                * 2.0 * np.pi * 1e-18
            print(f"  superficial: sigma = {sigma:.4g} C/m2 sobre {area:.3e} m2: "
                  f"{sigma * area / E_CHARGE:+.1f} e")

    @staticmethod
    def _areas(r, xp=np):
        rh = 0.5 * (r[:, 1:] + r[:, :-1])
        up = xp.concatenate([rh, r[:, -1:]], axis=1)
        lo = xp.concatenate([xp.zeros((r.shape[0], 1), dtype=r.dtype), rh], axis=1)
        return 0.5 * (up**2 - lo**2)

    @property
    def nunk(self):
        return 3 * (self.Nz + 1) * (self.N + 1)

    def liberar_cache(self):
        self._lu = None


def verificar_malla(m, tol=1e-9, verbose=True):
    """Comprueba z creciente, simetria especular de accesos/reservorios y
    perfiles de radio espejo. Devuelve dict con los errores."""
    z = m.z
    if not np.all(np.diff(z) > 0):
        raise AssertionError("z no es estrictamente creciente (nodos duplicados).")

    if not m.simetrica:
        out = {"simetrica": False, "Laccess_izq": m.Laccess, "Laccess_der": m.Laccess_R}
        if verbose:
            print(f"verificar_malla: z creciente OK; accesos distintos "
                  f"(izq {m.Laccess:g} nm, der {m.Laccess_R:g} nm): no se compara el espejo.")
        return out

    mL = z <= m.z_poro0 + 1e-12
    mR = z >= m.z_poro1 - 1e-12
    if mL.sum() != mR.sum():
        raise AssertionError("Distinto numero de nodos a izquierda y derecha.")
    # el espejo se compara solo en accesos+reservorios (el poro no es simetrico)
    err_z = float(np.max(np.abs((m.L - z[mR][::-1]) - z[mL])))
    sL = (m.rho[mL] - m.Rtip) / (m.Rres - m.Rtip)
    sR = (m.rho[mR][::-1] - m.Rbase) / (m.Rres - m.Rbase)
    err_rho = float(np.max(np.abs(sL - sR)))

    out = {"err_z_espejo_nm": err_z, "err_perfil_rho": err_rho,
           "nodos_izq": int(mL.sum()), "nodos_der": int(mR.sum())}
    if verbose:
        print(f"verificar_malla: |z_izq - (L - z_der_rev)|max = {err_z:.2e} nm, "
              f"|perfil_izq - perfil_der|max = {err_rho:.2e}, "
              f"nodos izq/der = {out['nodos_izq']}/{out['nodos_der']}")
    if err_z > tol * max(1.0, m.z_total) or err_rho > 1e-6:
        raise AssertionError(f"La malla no es especular: {out}")
    return out


# ----------------------------- interpolacion entre mallas -----------------------------
def extrapolar_solucion(u_vieja, m_viejo, m_nuevo):
    """Interpola la solucion de m_viejo a m_nuevo (lineal en z y en r/rho)."""
    U_old = np.asarray(u_vieja).reshape(m_viejo.Nz + 1, m_viejo.N + 1, 3)
    s_old = m_viejo.r / m_viejo.rho[:, None]
    s_new = m_nuevo.r / m_nuevo.rho[:, None]
    z_old = m_viejo.z

    U_new = np.empty((m_nuevo.Nz + 1, m_nuevo.N + 1, 3))
    for i, zi in enumerate(m_nuevo.z):
        i1 = int(np.clip(np.searchsorted(z_old, zi), 1, m_viejo.Nz))
        i0 = i1 - 1
        w = np.clip((zi - z_old[i0]) / (z_old[i1] - z_old[i0]), 0.0, 1.0)
        for v in range(3):
            p0 = np.interp(s_new[i], s_old[i0], U_old[i0, :, v])
            p1 = np.interp(s_new[i], s_old[i1], U_old[i1, :, v])
            U_new[i, :, v] = (1.0 - w) * p0 + w * p1

    U_new[..., 1:] = np.maximum(U_new[..., 1:], 1e-8)
    return _to_dev(U_new.ravel())


# ----------------------------- derivadas nodales -----------------------------
def _dz(f, m):
    g = _xp().empty_like(f)
    g[1:-1] = m.dz_wm * f[:-2] + m.dz_w0 * f[1:-1] + m.dz_wp * f[2:]
    g[0] = (f[1] - f[0]) * m.dz_ih0
    g[-1] = (f[-1] - f[-2]) * m.dz_ih1
    return g


def _dx(f):
    g = _xp().empty_like(f)
    g[:, 1:-1] = 0.5 * (f[:, 2:] - f[:, :-2])
    g[:, 0] = 0.0
    g[:, -1] = f[:, -1] - f[:, -2]
    return g


def _divergencia(PhiXi, PhiZ, dzc, wall=None):
    xp = _xp()
    n1, N = PhiXi.shape
    div = xp.empty((n1, N + 1))
    div[:, 0] = PhiXi[:, 0]
    div[:, 1:N] = PhiXi[:, 1:] - PhiXi[:, :-1]
    div[:, N] = -PhiXi[:, -1] if wall is None else wall - PhiXi[:, -1]
    div *= dzc
    div[1:-1] += PhiZ[1:] - PhiZ[:-1]
    return div


# ----------------------------- residuo PNP -----------------------------
def residuo(u, m, Vhat, sigh):
    """
    Poisson:  lap(phi) + (c+ - c- + X)/ell^2 = 0   (X = carga fija de la capa)
    Condiciones de borde axiales:
      z = z[0]  : phi = 0,    c+ = c- = 1
      z = z[-1] : phi = Vhat, c+ = c- = 1
    """
    xp = _xp()
    Nz, N = m.Nz, m.N
    U = u.reshape(Nz + 1, N + 1, 3)
    phi, cp_var, cm_var = U[..., 0], U[..., 1], U[..., 2]

    dzc, vol = m.dzc_gpu, m.vol_gpu
    axi, rzf, rf = m.axi_gpu, m.rzf_gpu, m.rf_gpu
    ihz, kz, Aface = m.ihz_gpu, m.kz_gpu, m.Aface_gpu

    phz = _dz(phi, m)
    phx = _dx(phi)
    dphi_x = phi[:, 1:] - phi[:, :-1]
    phz_f = 0.5 * (phz[:, 1:] + phz[:, :-1])
    dphi_z = phi[1:] - phi[:-1]
    phx_f = 0.5 * (phx[1:] + phx[:-1])

    R = xp.empty((Nz + 1, N + 1, 3))

    # Poisson (con carga volumetrica fija)
    PhiXi = rf * (axi * dphi_x - rzf * phz_f)
    PhiZ = (dphi_z * ihz - kz * phx_f) * Aface
    wall = sigh * m.sqrt_pared_gpu
    div = _divergencia(PhiXi, PhiZ, dzc, wall)
    R[..., 0] = (div + vol * (cp_var - cm_var + m.f_fijo * m.Xfix_gpu) / ELL2) / vol

    # Nernst-Planck (Scharfetter-Gummel en xi y en z)
    for k, (c, zs, D) in enumerate(((cp_var, +1.0, D_P), (cm_var, -1.0, D_M)), start=1):
        lc = xp.log(c)
        lz = _dz(lc, m)
        lx = _dx(lc)
        lz_f = 0.5 * (lz[:, 1:] + lz[:, :-1])
        c_f = 0.5 * (c[:, 1:] + c[:, :-1])
        Bp, Bm = _B2(zs * dphi_x)
        J = (
            D * axi * (Bp * c[:, :-1] - Bm * c[:, 1:])
            + D * rzf * c_f * (lz_f + zs * phz_f)
        )
        PhiXi = rf * J

        lx_f = 0.5 * (lx[1:] + lx[:-1])
        cc = 0.5 * (c[1:] + c[:-1])
        Bp, Bm = _B2(zs * dphi_z)
        qz = (
            D * ihz * (Bp * c[:-1] - Bm * c[1:])
            + D * kz * cc * (lx_f + zs * phx_f)
        )
        PhiZ = qz * Aface
        R[..., k] = _divergencia(PhiXi, PhiZ, dzc) / vol

    # Dirichlet en los extremos
    R[0, :, 0] = phi[0]
    R[0, :, 1] = cp_var[0] - 1.0
    R[0, :, 2] = cm_var[0] - 1.0

    R[-1, :, 0] = phi[-1] - Vhat
    R[-1, :, 1] = cp_var[-1] - 1.0
    R[-1, :, 2] = cm_var[-1] - 1.0
    return R.ravel()


# ----------------------------- Jacobiano disperso -----------------------------
class _PatronJ:
    """Patron de dispersion y coloreado del Jacobiano (se calcula una vez por malla)."""

    def __init__(self, m, delta=1e-6, seed=0):
        Nz, N = m.Nz, m.N
        n = m.nunk
        self.n = n
        idx = np.arange(n).reshape(Nz + 1, N + 1, 3)

        rng = np.random.default_rng(seed)
        U = np.ones((Nz + 1, N + 1, 3))
        U[..., 0] = rng.uniform(-1.0, 1.0, (Nz + 1, N + 1))
        U[..., 1:] = rng.uniform(0.8, 1.2, (Nz + 1, N + 1, 2))
        u = _to_dev(U.ravel())
        Vhat, sigh = 0.7, 0.3
        F0 = residuo(u, m, Vhat, sigh)

        s = 5
        rows_l, cols_l = [], []
        dmax = 0
        for a in range(s):
            for b in range(s):
                for v in range(3):
                    col = idx[a::s, b::s, v].ravel()
                    if col.size == 0:
                        continue
                    up = u.copy()
                    up[_to_dev(col)] += delta
                    dF = _to_cpu(residuo(up, m, Vhat, sigh) - F0)
                    r = np.flatnonzero(dF)
                    node = r // 3
                    i, j = node // (N + 1), node % (N + 1)
                    ic = a + s * np.rint((i - a) / s).astype(np.int64)
                    jc = b + s * np.rint((j - b) / s).astype(np.int64)
                    ok = (ic >= 0) & (ic <= Nz) & (jc >= 0) & (jc <= N)
                    if not np.all(ok):
                        raise RuntimeError("Patron del Jacobiano inconsistente.")
                    if r.size:
                        dmax = max(dmax, int(np.abs(i - ic).max()), int(np.abs(j - jc).max()))
                    rows_l.append(r)
                    cols_l.append(idx[ic, jc, v])
        rows = np.concatenate(rows_l)
        cols = np.concatenate(cols_l)
        order = np.lexsort((rows, cols))
        rows, cols = rows[order], cols[order]
        self.nnz = rows.size
        self.indices = rows.astype(np.int32)
        self.indptr = np.searchsorted(cols, np.arange(n + 1)).astype(np.int32)

        self.ancho = max(dmax, 1)
        s2 = 2 * self.ancho + 1
        node = cols // 3
        v_c = cols % 3
        i_c, j_c = node // (N + 1), node % (N + 1)
        color = ((i_c % s2) * s2 + (j_c % s2)) * 3 + v_c
        self.colores = []
        orden_color = np.argsort(color, kind="stable")
        cortes = np.searchsorted(color[orden_color], np.arange(s2 * s2 * 3 + 1))
        for k in range(s2 * s2 * 3):
            sel = orden_color[cortes[k]:cortes[k + 1]]
            a, rest = divmod(k, s2 * 3)
            b, v = divmod(rest, 3)
            colk = idx[a::s2, b::s2, v].ravel()
            if colk.size == 0 or sel.size == 0:
                continue
            sel = np.sort(sel)
            self.colores.append((_to_dev(colk), sel, rows[sel], _to_dev(rows[sel]),
                                 cols[sel]))
        self.n_colores = len(self.colores)

    def matriz(self, u, F0, m, Vhat, sigh, delta):
        # Paso de diferencias finitas RELATIVO para las concentraciones (con carga
        # fija alta c- puede ser ~1e-10 y un paso absoluto de 1e-6 arruina el
        # Jacobiano); para phi, paso ~ delta*(1+|phi|).
        u_cpu = _to_cpu(u)
        h_cpu = np.where(np.arange(u_cpu.size) % 3 == 0,
                         delta * (1.0 + np.abs(u_cpu)),
                         delta * np.maximum(np.abs(u_cpu), 1e-12))
        h_dev = _to_dev(h_cpu)
        data = np.empty(self.nnz)
        for col_dev, sel, filas, filas_dev, cols_sel in self.colores:
            up = u.copy()
            up[col_dev] += h_dev[col_dev]
            dF = residuo(up, m, Vhat, sigh) - F0
            data[sel] = _to_cpu(dF[filas_dev]) / h_cpu[cols_sel]
        return sp.csc_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))


def jacobiano(u, F0, m, Vhat, sigh, delta=1e-6):
    if m._patron is None:
        m._patron = _PatronJ(m, delta)
    return m._patron.matriz(u, F0, m, Vhat, sigh, delta)


# ----------------------------- Newton -----------------------------
def _newton(u, m, Vhat, sigh, tol, maxit, verbose, usar_lu_previo):
    xp = _xp()
    lu = m._lu if usar_lu_previo else None
    uso_viejo = lu is not None
    paso_prev = None
    for it in range(maxit):
        F0 = residuo(u, m, Vhat, sigh)
        if not bool(xp.all(xp.isfinite(F0))):
            if verbose:
                print("    [Newton] Residuo no finito.", flush=True)
            return u, False, uso_viejo

        fresco = False
        try:
            if lu is None:
                J = jacobiano(u, F0, m, Vhat, sigh)
                lu = spla.splu(J)
                m._lu = lu
                fresco = True
            du_cpu = lu.solve(-_to_cpu(F0))
        except Exception as exc:
            if verbose:
                print(f"    [Newton] Fallo en Jacobiano/solver: {exc!r}")
                traceback.print_exc()
            m._lu = None
            return u, False, uso_viejo

        if not np.all(np.isfinite(du_cpu)):
            if verbose:
                print("    [Newton] Paso du no finito.", flush=True)
            m._lu = None
            return u, False, uso_viejo
        du = _to_dev(du_cpu)

        D3 = du.reshape(-1, 3)
        U3 = u.reshape(-1, 3)

        d0_max = float(xp.max(xp.abs(D3[:, 0])))
        alpha = min(1.0, 3.0 / max(d0_max, 1e-30))
        neg = D3[:, 1:] < 0.0
        if bool(xp.any(neg)):
            ratio = U3[:, 1:][neg] / -D3[:, 1:][neg]
            alpha = min(alpha, float(0.7 * xp.min(ratio)))

        if not np.isfinite(alpha) or alpha <= 0.0:
            if verbose:
                print(f"    [Newton] alpha invalido: {alpha}", flush=True)
            return u, False, uso_viejo

        u = u + alpha * du
        Unew = u.reshape(-1, 3)
        if float(xp.min(Unew[:, 1:])) <= 0.0:
            if verbose:
                print("    [Newton] Concentracion no positiva.", flush=True)
            return u, False, uso_viejo

        err_phi = float(xp.max(xp.abs(D3[:, 0])))
        # error relativo con piso 1e-6: concentraciones << C0 (co-iones casi excluidos)
        # no deben dominar el criterio de convergencia
        err_c = float(xp.max(xp.abs(D3[:, 1:]) / xp.maximum(U3[:, 1:], 1e-6)))
        err = max(err_phi, err_c)
        paso = err * alpha
        if verbose:
            tag = "J" if fresco else "LU"
            print(f"    it {it:2d}  |du|rel={paso:.2e}  alpha={alpha:.3f}  [{tag}]", flush=True)
        if paso < tol and alpha == 1.0:
            return u, True, uso_viejo

        if (not fresco) and paso_prev is not None and paso > 0.3 * paso_prev:
            lu = None
        paso_prev = paso
    return u, False, uso_viejo


def newton(u, m, Vhat, sigh, tol=1e-7, maxit=40, verbose=False):
    u_ini = u
    u, ok, uso_viejo = _newton(u_ini, m, Vhat, sigh, tol, maxit, verbose, True)
    if not ok and uso_viejo:
        if verbose:
            print("    [Newton] Reintento con Jacobiano fresco.", flush=True)
        m._lu = None
        u, ok, _ = _newton(u_ini, m, Vhat, sigh, tol, maxit, verbose, False)
    return u, ok


def estado_inicial(m):
    xp = _xp()
    U = xp.ones((m.Nz + 1, m.N + 1, 3))
    U[..., 0] = 0.0
    return U.ravel()


def resolver(m, V_volt, sigma, u0=None, verbose=False):
    """V_volt se aplica en el extremo DERECHO (el izquierdo esta a tierra)."""
    sigh = sigma * SIGHAT
    if u0 is None:
        m._lu = None
        u = estado_inicial(m)
        # Continuacion adaptativa simultanea en carga superficial y volumetrica a V = 0:
        # si un paso no converge se reduce (biseccion), si converge se agranda.
        f, paso = 0.0, 0.1
        m.f_fijo = 0.0
        u, ok = newton(u, m, 0.0, 0.0, verbose=verbose)
        if not ok:
            m.f_fijo = 1.0
            return u, False
        while f < 1.0 - 1e-12:
            f_new = min(1.0, f + paso)
            m.f_fijo = f_new
            u_try, ok = newton(u, m, 0.0, f_new * sigh, verbose=verbose)
            if ok:
                u, f = u_try, f_new
                paso = min(0.25, paso * 1.5)
            else:
                paso *= 0.5
                if paso < 1.0 / 64:
                    m.f_fijo = 1.0
                    return u, False
        m.f_fijo = 1.0
    else:
        m.f_fijo = 1.0
        u = u0.copy()

    Vhat = V_volt / VT
    return newton(u, m, Vhat, sigh, verbose=verbose)


# ----------------------------- corriente -----------------------------
def corriente(u, m):
    """Corriente (A), positiva del electrodo derecho (V) hacia el izquierdo (tierra).
    Devuelve (promedio dentro del poro, valor en cada cara axial)."""
    xp = _xp()
    Nz, N = m.Nz, m.N
    U = u.reshape(Nz + 1, N + 1, 3)
    phi, cp_var, cm_var = U[..., 0], U[..., 1], U[..., 2]
    tot = xp.zeros(Nz)

    phx = _dx(phi)
    dphi_z = phi[1:] - phi[:-1]
    phx_f = 0.5 * (phx[1:] + phx[:-1])

    ihz, kz, Aface = m.ihz_gpu, m.kz_gpu, m.Aface_gpu

    for c, zs, D, sg in ((cp_var, +1.0, D_P, +1), (cm_var, -1.0, D_M, -1)):
        lx = _dx(xp.log(c))
        lx_f = 0.5 * (lx[1:] + lx[:-1])
        cc = 0.5 * (c[1:] + c[:-1])
        Bp, Bm = _B2(zs * dphi_z)
        qz = (
            D * ihz * (Bp * c[:-1] - Bm * c[1:])
            + D * kz * cc * (lx_f + zs * phx_f)
        )
        tot += sg * (qz * Aface).sum(axis=1)

    I_faces = SIGNO_I * 2.0 * np.pi * F * D0 * C0 * 1e-9 * _to_cpu(tot)
    zmid = 0.5 * (m.z[:-1] + m.z[1:])
    mask = (zmid >= m.z_poro0) & (zmid <= m.z_poro1)
    if not np.any(mask):
        raise RuntimeError("No hay caras axiales dentro del nanoporo.")
    return I_faces[mask].mean(), I_faces


# ----------------------------- Curva I-V -----------------------------
def _malla_gruesa(m):
    return Malla(
        Rtip=m.Rtip, Rbase=m.Rbase, L_h=m.L_h, L=m.L,
        Rres=m.Rres, Laccess=m.Laccess,
        Laccess_R=m.Laccess_R,
        Nz_access_R=max(20, m.Nz_access_R // 2),
        dz0_access_R=m.dz0_access_R * 2.0,
        Nz_poro=max(40, m.Nz_poro // 2),
        Nz_res=max(20, m.Nz_res // 2),
        Nz_access=max(20, m.Nz_access // 2),
        Neta=max(16, m.N // 2),
        dz0=m.dz0 * 2.0,
        dw0=m.dw0 * 2.0,
        dz0_access=m.dz0_access * 2.0,
        dz0_res=m.dz0_res * 2.0,
        malla_poro=m.malla_poro, A_mon=m.A_mon, B_mon=m.B_mon,
        modo_carga=m.modo_carga, rho_vol=m.rho_vol, z_vol=m.z_vol, S_vol=m.S_vol,
        rho_rad=m.rho_rad, z_rad=m.z_rad,
        sigma_en_zona=m.sigma_en_zona, dz_vol=m.dz_vol * 2.0,
    )


def curva_IV(m_final, V_list, sigma, verbose=True, dV_max=0.1,
             usar_refinamiento=False, verbose_newton=True):
    """
    Curva I-V (V a la derecha, tierra a la izquierda). Con usar_refinamiento=True
    se resuelve primero en una malla gruesa (misma capa de carga) y se interpola.
    Si un voltaje no converge, se devuelven los puntos calculados (resto = NaN).
    """
    m = m_final
    u0 = None

    if usar_refinamiento:
        print("\n--- [Paso 1/2] Resolviendo en malla gruesa preliminar ---")
        m_gruesa = _malla_gruesa(m_final)
        u0_gruesa, ok = resolver(m_gruesa, 0.0, sigma, verbose=verbose_newton)
        if not ok:
            print("-> Aviso: la malla gruesa fallo. Se intenta directamente con la malla final...")
        else:
            print("--- [Paso 2/2] Migrando solucion a la malla fina final ---")
            u0_interp = extrapolar_solucion(_to_cpu(u0_gruesa), m_gruesa, m_final)
            u0_try, ok = resolver(m_final, 0.0, sigma, u0=u0_interp, verbose=verbose_newton)
            if ok:
                u0 = u0_try
            else:
                print("-> Aviso: el refinamiento directo fallo. Se usa la continuacion estandar...")
        m_gruesa.liberar_cache()

    V_list = np.asarray(V_list, dtype=float)
    if V_list.ndim != 1 or V_list.size == 0:
        raise ValueError("V_list debe ser un vector no vacio.")

    I = np.full(V_list.shape, np.nan)
    sol = {}

    if u0 is None:
        u0, ok = resolver(m, 0.0, sigma, verbose=verbose_newton)
        if not ok:
            raise RuntimeError("No converge el equilibrio V=0. Revisar malla, sigma, rho_vol y dimensiones.")

    I0_raw, _ = corriente(u0, m)

    for k, V in enumerate(V_list):
        if abs(V) < 1e-14:
            I[k] = 0.0
            sol[float(V)] = _to_cpu(u0).copy()

    for signo in (+1, -1):
        indices = [
            k for k in np.argsort(np.abs(V_list))
            if signo * V_list[k] > 1e-14
        ]
        u = u0.copy()
        Vprev = 0.0
        fallo = False
        for k in indices:
            V = float(V_list[k])
            nsub = max(1, int(np.ceil(abs(V - Vprev) / dV_max)))
            trayectoria = np.linspace(Vprev, V, nsub + 1)[1:]
            for Vi in trayectoria:
                u_new, ok = resolver(m, float(Vi), sigma, u0=u, verbose=verbose_newton)
                if not ok:
                    print(f"  !! No converge en V={Vi:+.4f} V; se conservan los puntos previos.",
                          flush=True)
                    fallo = True
                    break
                u = u_new
            if fallo:
                break
            Vprev = V
            sol[V] = _to_cpu(u).copy()
            I_raw, _ = corriente(u, m)
            I[k] = I_raw - I0_raw
            if verbose:
                print(f"  V={V:+.3f} V    I={I[k] * 1e9:+.6f} nA", flush=True)

    m.liberar_cache()
    return I, sol