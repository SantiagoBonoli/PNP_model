# -*- coding: utf-8 -*-
"""
MALLA 2D axisimetrica (r, z) para nanoporo con reservorios, boca en ESCALON real.
Unidades: nm. Sin desbordamiento radial en membrana ni volumen ocluido.
"""
import warnings
import numpy as np

# IMPORTANTE: Forzar backend no interactivo para evitar bloqueos en Wayland
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from scipy.optimize import brentq

HAS_GPU = False
USAR_GPU = False
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    HAS_GPU = True
except Exception:
    import numpy as cp


def usar_gpu(flag):
    """Llamar ANTES de construir la Malla."""
    global USAR_GPU
    USAR_GPU = bool(flag) and HAS_GPU
    print(f"--- Calculo en {'GPU (CuPy)' if USAR_GPU else 'CPU (NumPy)'} ---")


def _xp():
    return cp if USAR_GPU else np


def _cpu(a):
    return cp.asnumpy(a) if USAR_GPU else np.asarray(a)


def _dev(a):
    return cp.asarray(a) if USAR_GPU else np.asarray(a)


# ----------------------------- constantes -----------------------------
F = 96485.33
R_GAS = 8.314462
T = 298.15
N_A = 6.02214076e23
E_CHARGE = 1.602176634e-19
VT = R_GAS * T / F
EPS = 78.5 * 8.8541878128e-12
C0 = 100.0                              # mol/m3 = 0.1 M
D_K, D_CL = 1.957e-9, 2.032e-9
KAPPA_NE = F**2 * C0 * (D_K + D_CL) / (R_GAS * T)
KAPPA_OBJETIVO = KAPPA_NE               # S/m, consistente con Nernst-Einstein / COMSOL (~1.4981 S/m)
FACT_D = KAPPA_OBJETIVO / KAPPA_NE
D0 = D_K * FACT_D
D_P = 1.0
D_M = D_CL / D_K
ELL2 = EPS * R_GAS * T / (F**2 * C0) * 1e18   # nm2; lambda_D^2 = ELL2/2
SIGHAT = F / (EPS * R_GAS * T) * 1e-9         # (C/m2) -> (1/nm)
SIGNO_I = -1.0


# ----------------------------- utilidades 1D -----------------------------
def _malla_1d(x0, x1, hfun, hmin, puntos=()):
    L = x1 - x0
    d = np.geomspace(max(0.01 * hmin, 1e-9 * L), L, 400)
    pts = [np.linspace(x0, x1, 6001), x0 + d, x1 - d]
    for p in puntos:
        if x0 < p < x1:
            pts += [p + d, p - d]
    xs = np.unique(np.clip(np.concatenate(pts), x0, x1))
    inv = 1.0 / hfun(xs)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv[1:] + inv[:-1]) * np.diff(xs))])
    n = max(2, int(np.ceil(cum[-1] * (1.0 - 1e-9))))
    x = np.interp(np.linspace(0.0, cum[-1], n + 1), cum, xs)
    x[0], x[-1] = x0, x1
    return x


def _nodos_geom(x0, x1, n, dx0):
    largo = float(x1 - x0)
    if dx0 * n >= largo * (1.0 - 1e-12):
        return np.linspace(x0, x1, n + 1)
    hi = min(50.0, float(np.exp(600.0 / n)))
    f = lambda q: dx0 * np.expm1(n * np.log(q)) / (q - 1.0) - largo
    q = brentq(f, 1.0 + 1e-9, hi)
    d = np.concatenate([[0.0], np.cumsum(dx0 * q ** np.arange(n))])
    d *= largo / d[-1]
    return x0 + d


_GX = np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])
_GW = np.array([5.0, 8.0, 5.0]) / 18.0


def _gauss_cells(za, zb, f):
    zm = 0.5 * (za + zb)[:, None] + 0.5 * (zb - za)[:, None] * _GX[None, :]
    return (zb - za) * np.sum(f(zm) * _GW, axis=1)


# ----------------------------- malla -----------------------------
class Malla:
    def __init__(
        self, Rtip, Rbase, L_h, L=12000.0, Rres=25000.0, Lres=None,
        N=1,
        h_boca=None,
        q_z=1.10,
        alpha_poro=0.5,
        h_max_poro=None,
        h_borde=None,
        h_max_res=None,
        dw_pared=None,
        q_xi=1.15,
        dxi_max=0.1,
        q_r=1.10,
        refinar=2.0,
        modo_carga="superficial",
        rho_vol=0.0, z_vol=None, S_vol=None, rho_rad=None, z_rad=None,
        sigma_en_zona=False,
        sigma_membrana=0.0,
    ):
        if min(Rtip, Rbase, L) <= 0 or L_h < 0:
            raise ValueError("Rtip, Rbase, L > 0 y L_h >= 0.")
        if Rres <= max(Rtip, Rbase):
            raise ValueError("Rres debe ser mayor que Rtip y Rbase.")
        if modo_carga not in ("superficial", "vol_z", "vol_radial", "vol_combinado"):
            raise ValueError("modo_carga invalido.")
        if modo_carga in ("vol_z", "vol_combinado") and z_vol is None:
            raise ValueError(f"modo '{modo_carga}' requiere z_vol=(za, zb).")
        if modo_carga in ("vol_radial", "vol_combinado") and (S_vol is None or S_vol <= 0):
            raise ValueError(f"modo '{modo_carga}' requiere S_vol > 0 (nm).")
        if modo_carga == "vol_radial" and z_vol is None:
            z_vol = (0.0, L)
        f = float(refinar)

        self.Rtip, self.Rbase, self.L_h, self.L = float(Rtip), float(Rbase), float(L_h), float(L)
        self.Rres = float(Rres)
        self.Lres = float(Rres if Lres is None else Lres)
        self.sigma_membrana = float(sigma_membrana)
        Rmin, Rmax = min(Rtip, Rbase), max(Rtip, Rbase)
        lamD = np.sqrt(ELL2 / 2.0)
        self.N = float(N)

        def _tramo(zz):
            a, b = max(float(zz[0]), 0.0), min(float(zz[1]), float(L))
            if not b > a:
                raise ValueError("Zona fuera de [0, L].")
            return (a, b)

        zonas = []
        self.modo_carga, self.sigma_en_zona = modo_carga, bool(sigma_en_zona)
        rv = float(rho_vol)
        rr = rv if rho_rad is None else float(rho_rad)
        S = None if S_vol is None else float(S_vol)
        if z_vol is not None:
            z_vol = _tramo(z_vol)
        if z_rad is not None:
            z_rad = _tramo(z_rad)
        if modo_carga == "vol_z":
            zonas.append((z_vol[0], z_vol[1], rv, None))
        elif modo_carga == "vol_radial":
            zonas.append((z_vol[0], z_vol[1], rv, S))
        elif modo_carga == "vol_combinado":
            za, zb = z_vol
            zonas.append((za, zb, rv, None))
            if z_rad is None:
                if za > 0.0:
                    zonas.append((0.0, za, rr, S))
                if zb < L:
                    zonas.append((zb, float(L), rr, S))
            else:
                zonas.append((z_rad[0], z_rad[1], rr, S))
        self.zonas = [z for z in zonas if z[2] != 0.0]
        self.hay_vol = len(self.zonas) > 0

        hb = (0.1 if h_boca is None else float(h_boca)) / f
        if h_boca is None:
            hb = min(hb * f, max(0.02, 0.02 * Rmin)) / f
        hbord = (max(hb * f, 0.3) if h_borde is None else float(h_borde)) / f
        hmaxp = (L / 150.0 if h_max_poro is None else float(h_max_poro)) / f
        hmaxr = (0.1 * min(self.Rres, self.Lres) if h_max_res is None else float(h_max_res)) / f
        g = q_z - 1.0
        self.h_boca = hb

        prf = lambda z: self.perfil(z)
        bordes = sorted(set(x for (a, b, _, _) in self.zonas for x in (a, b) if 0.0 < x < L))

        feat = [(0.0, hb), (float(L), hb)] + [(e, hbord) for e in bordes]

        def hpor(z):
            h = np.minimum(hmaxp, alpha_poro * prf(z))
            for zk, hk in feat:
                h = np.minimum(h, hk + g * np.abs(z - zk))
            return h

        espej = (abs(Rtip - Rbase) < 1e-12 and
                 np.allclose(sorted(bordes), sorted(L - b for b in bordes), atol=1e-9))
        self.simetrica_poro = bool(espej)
        bp = sorted(set([0.0, float(L)] + bordes + ([0.5 * L] if espej else [])))
        if espej:
            bp = [b for b in bp if b <= 0.5 * L + 1e-12]
        zp = [0.0]
        for a, b in zip(bp[:-1], bp[1:]):
            zp.extend(_malla_1d(a, b, hpor, hb, puntos=[0.0, L] + bordes)[1:])
        zp = np.array(zp)
        if espej:
            zp = np.concatenate([zp, L - zp[-2::-1]])
        zp[0], zp[-1] = 0.0, float(L)

        hres = lambda d: np.minimum(hmaxr, hb + g * d)
        dres = _malla_1d(0.0, self.Lres, hres, hb)
        zR = L + dres
        zL = -dres[::-1]
        self.zf = np.concatenate([zL[:-1], zp, zR[1:]])
        self.k0 = len(zL) - 1
        self.kL = self.k0 + len(zp) - 1
        zf = self.zf
        self.Nzc = len(zf) - 1
        self.zc = 0.5 * (zf[1:] + zf[:-1])
        self.dz = np.diff(zf)
        nL, nP = self.k0, self.kL - self.k0
        self.tipo = np.zeros(self.Nzc, int)
        self.tipo[nL:nL + nP] = 1
        self.tipo[nL + nP:] = 2

        dw = 0.3 * lamD if dw_pared is None else float(dw_pared)
        self.dxi_w = (dw / f) / Rmax
        hs = lambda s: np.minimum(dxi_max / f, self.dxi_w + (q_xi - 1.0) * s)
        s = _malla_1d(0.0, 1.0, hs, self.dxi_w)
        self.xi = 1.0 - s[::-1]
        self.xi[0], self.xi[-1] = 0.0, 1.0
        self.Nxi = len(self.xi) - 1

        def n_out(Rs):
            h0 = self.dxi_w * Rs
            return int(np.ceil(np.log1p((self.Rres - Rs) * (q_r - 1.0) / h0) / np.log(q_r)))

        self.Nout = max(4, n_out(Rtip), n_out(Rbase))
        self.Nr = self.Nxi + self.Nout
        Nr, Nxi = self.Nr, self.Nxi

        def radios_res(Rs):
            h0 = self.dxi_w * Rs
            return np.concatenate([Rs * self.xi,
                                   _nodos_geom(Rs, self.Rres, self.Nout, h0)[1:]])

        rL, rR = radios_res(Rtip), radios_res(Rbase)

        rho_c = self.perfil(self.zc)
        drho_c = self.dperfil(self.zc)
        rm = np.empty((self.Nzc, Nr + 1))
        sl = np.zeros((self.Nzc, Nr + 1))
        rm[self.tipo == 0] = rL
        rm[self.tipo == 2] = rR
        ip = self.tipo == 1
        rm[ip, :Nxi + 1] = rho_c[ip, None] * self.xi[None, :]

        # Correccion: en la membrana (j > Nxi), interpolar entre rho(z) y Rres
        # sin desbordar Rres ni generar celdas solapadas/ocluidas.
        eta_out = (rL[Nxi + 1:] - Rtip) / (self.Rres - Rtip)
        rm[ip, Nxi + 1:] = rho_c[ip, None] + (self.Rres - rho_c[ip, None]) * eta_out[None, :]

        sl[ip, :Nxi + 1] = drho_c[ip, None] * self.xi[None, :]
        self.rm, self.slope = rm, sl
        self.rc = 0.5 * (rm[:, 1:] + rm[:, :-1])
        act = np.ones((self.Nzc, Nr), bool)
        act[ip, Nxi:] = False
        self.act = act

        xi2 = 0.5 * (self.xi[1:]**2 - self.xi[:-1]**2)
        V = np.empty((self.Nzc, Nr))
        r2 = 0.5 * (rm[:, 1:]**2 - rm[:, :-1]**2)
        V[:] = r2 * self.dz[:, None]
        irho2 = _gauss_cells(zf[:-1][ip], zf[1:][ip], lambda zz: self.perfil(zz)**2)
        V[ip, :Nxi] = xi2[None, :] * irho2[:, None]
        V[ip, Nxi:] = 0.0
        self.V = V

        A = np.zeros((self.Nzc + 1, Nr))
        for k in range(self.Nzc + 1):
            if k < self.k0:
                rr_ = rL
            elif k > self.kL:
                rr_ = rR
            else:
                A[k, :Nxi] = xi2 * self.perfil(zf[k])**2
                continue
            A[k] = 0.5 * (rr_[1:]**2 - rr_[:-1]**2)
        self.A = A

        hz = np.empty(self.Nzc + 1)
        hz[1:-1] = self.zc[1:] - self.zc[:-1]
        hz[0], hz[-1] = 0.5 * self.dz[0], 0.5 * self.dz[-1]
        self.hz = hz

        sf = np.zeros((self.Nzc + 1, Nr))
        xic = 0.5 * (self.xi[1:] + self.xi[:-1])
        for k in range(self.k0 + 1, self.kL):
            sf[k, :Nxi] = self.dperfil(zf[k]) * xic
        # En las bocas k0 y kL, pendiente efectiva ponderada entre reservorio (0) y poro
        w0_poro = self.dz[self.k0] / (self.dz[self.k0 - 1] + self.dz[self.k0])
        wL_poro = self.dz[self.kL - 1] / (self.dz[self.kL - 1] + self.dz[self.kL])
        sf[self.k0, :Nxi] = w0_poro * self.dperfil(0.0) * xic
        sf[self.kL, :Nxi] = wL_poro * self.dperfil(self.L) * xic
        self.sf = sf

        Aw = np.zeros((self.Nzc, Nr))
        zi0, zi1 = zf[:-1][ip], zf[1:][ip]
        awp = _gauss_cells(zi0, zi1,
                           lambda zz: self.perfil(zz) * np.sqrt(1.0 + self.dperfil(zz)**2))
        zcp = self.zc[ip]
        libre = np.ones(zcp.size)
        if self.hay_vol and not self.sigma_en_zona:
            for (a, b, _, _) in self.zonas:
                libre[(zcp > a) & (zcp < b)] = 0.0
        Aw[ip, Nxi - 1] = awp * libre
        self.Aw_sigma = Aw

        Am = np.zeros((self.Nzc, Nr))
        if self.sigma_membrana != 0.0:
            Am[self.k0 - 1, Nxi:] = 0.5 * (rL[Nxi + 1:]**2 - rL[Nxi:-1]**2)
            Am[self.kL, Nxi:] = 0.5 * (rR[Nxi + 1:]**2 - rR[Nxi:-1]**2)
        self.Aw_mem = Am

        X = np.zeros((self.Nzc, Nr))
        for (a, b, rc_, Sz) in self.zonas:
            enz = ip & (self.zc > a) & (self.zc < b)
            if not enz.any():
                continue
            ii = np.flatnonzero(enz)
            if Sz is None:
                frac = np.ones((ii.size, Nxi))
            else:
                za, zb = zf[ii], zf[ii + 1]
                num = np.zeros((ii.size, Nxi))
                for xg, wg in zip(_GX, _GW):
                    zz = 0.5 * (za + zb) + 0.5 * (zb - za) * xg
                    rh = self.perfil(zz)[:, None]
                    lo = np.maximum(rh * self.xi[None, :-1], rh - Sz)
                    up = rh * self.xi[None, 1:]
                    num += wg * 0.5 * np.maximum(up**2 - lo**2, 0.0)
                den = np.maximum(V[ii, :Nxi] / self.dz[ii, None], 1e-300)
                frac = np.clip(num / den, 0.0, 1.0)
            X[ii, :Nxi] += (rc_ / (F * C0)) * frac
        self.Xfix = X

        self._preparar_operadores()

    def perfil(self, z):
        z = np.asarray(z, float)
        x = np.clip(z / self.L, 0.0, 1.0)
        if abs(self.L_h) < 1e-8:
            gg = x
        else:
            gg = (1.0 - np.exp(-(self.L_h * x)**self.N)) / (1.0 - np.exp(-self.L_h**self.N))
        return self.Rtip + (self.Rbase - self.Rtip) * gg

    def dperfil(self, z):
        z = np.asarray(z, float)
        x = np.clip(z / self.L, 0.0, 1.0)
        if abs(self.L_h) < 1e-8:
            dg = np.ones_like(x) / self.L
        else:
            dg = (
                self.N * self.L_h**self.N * x**(self.N - 1)
                * np.exp(-(self.L_h * x)**self.N)
                / (1.0 - np.exp(-self.L_h**self.N))
                / self.L
            )
        return (self.Rbase - self.Rtip) * dg

    def _preparar_operadores(self):
        Nzc, Nr, Nxi = self.Nzc, self.Nr, self.Nxi
        act, A, rm, rc = self.act, self.A, self.rm, self.rc
        actp = np.vstack([act[:1], act, act[-1:]])
        abierta_z = actp[:-1] & actp[1:]

        # Correccion metrica axial: (1 + sf^2) en las caras no ortogonales del poro
        wz = np.where(abierta_z, A * (1.0 + self.sf**2) / self.hz[:, None], 0.0)

        abierta_r = act[:, :-1] & act[:, 1:]
        drc = rc[:, 1:] - rc[:, :-1]
        sm = self.slope[:, 1:-1]
        wr = np.where(abierta_r,
                      self.dz[:, None] * rm[:, 1:-1] * (1.0 + sm**2) / np.where(drc > 0, drc, 1.0),
                      0.0)
        crz = np.where(abierta_r, -self.dz[:, None] * rm[:, 1:-1] * sm, 0.0)
        crr = np.where(abierta_z, -A * self.sf, 0.0)

        hm = self.zc[1:-1] - self.zc[:-2]
        hp = self.zc[2:] - self.zc[1:-1]
        dzm = (-hp / (hm * (hm + hp)))[:, None]
        dz0 = ((hp - hm) / (hm * hp))[:, None]
        dzp = (hm / (hp * (hp + hm)))[:, None]
        J = np.arange(Nr)[None, :].repeat(Nzc, 0)
        jl = np.where(self.tipo == 1, Nxi - 1, Nr - 1)[:, None]
        JP = np.minimum(J + 1, jl)
        JM = np.maximum(J - 1, 0)
        idr = 1.0 / np.maximum(np.take_along_axis(rc, JP, 1) - np.take_along_axis(rc, JM, 1), 1e-300)
        idr = np.where(JP == JM, 0.0, idr)
        d = {k: _dev(v) for k, v in dict(
            wz=wz, wr=wr, crz=crz, crr=crr, dzm=dzm, dz0=dz0, dzp=dzp, JP=JP, JM=JM, idr=idr,
            V=self.V, act=act, Xfix=self.Xfix, Asig=self.Aw_sigma, Amem=self.Aw_mem).items()}
        d["ih0"] = float(1.0 / (self.zc[1] - self.zc[0]))
        d["ih1"] = float(1.0 / (self.zc[-1] - self.zc[-2]))
        self.g = d

    @property
    def nunk(self):
        return int(self.act.sum()) * 3

    def exportar_mallado(self, nombre_archivo="malla_exportada.npz"):
        np.savez(
            nombre_archivo,
            zf=_cpu(self.zf),
            zc=_cpu(self.zc),
            xi=_cpu(self.xi),
            rm=_cpu(self.rm),
            rc=_cpu(self.rc),
            act=_cpu(self.act),
            V=_cpu(self.V),
            Xfix=_cpu(self.Xfix)
        )
        print(f"--- Mallado exportado exitosamente a '{nombre_archivo}' ---")

    def visualizar_mallado(self, archivo_salida="malla_esquema.png"):
        """Genera un grafico de 2 paneles: (1) Dominio completo con reservorios y (2) Zoom en el poro.
        Solo dibuja las celdas activas de fluido con cuadrilateros curvilineos exactos
        (sin celdas falsas en la membrana ni volumen ocluido)."""
        rm_cpu = _cpu(self.rm)
        zf_cpu = _cpu(self.zf)
        act_cpu = _cpu(self.act)

        Nzc, Nr = act_cpu.shape
        lines_fluid = []

        for i in range(Nzc):
            z0, z1 = zf_cpu[i], zf_cpu[i + 1]
            es_poro = (self.tipo[i] == 1)
            if es_poro:
                rho0 = float(self.perfil(z0))
                rho1 = float(self.perfil(z1))
            for j in range(Nr):
                if not act_cpu[i, j]:
                    continue
                if es_poro:
                    r0_a, r1_a = rho0 * self.xi[j], rho0 * self.xi[j + 1]
                    r0_b, r1_b = rho1 * self.xi[j], rho1 * self.xi[j + 1]
                else:
                    r0_a = r0_b = rm_cpu[i, j]
                    r1_a = r1_b = rm_cpu[i, j + 1]

                for s in (+1.0, -1.0):
                    p00 = (z0, s * r0_a)
                    p10 = (z1, s * r0_b)
                    p11 = (z1, s * r1_b)
                    p01 = (z0, s * r1_a)
                    lines_fluid.extend([
                        (p00, p10),
                        (p10, p11),
                        (p11, p01),
                        (p01, p00),
                    ])

        fig, axs = plt.subplots(1, 2, figsize=(14, 5.5))
        z_line = np.linspace(0, self.L, 1000)
        r_line = self.perfil(z_line)

        ventanas = [
            (axs[0], float(zf_cpu[0]) - 0.05 * self.Lres, float(zf_cpu[-1]) + 0.05 * self.Lres,
             1.15 * self.Rres, "Dominio Completo (Reservorios + Poro)", 0.15),
            (axs[1], -100, 100,
             2.5 * self.Rtip, "Zoom en la Boca Izquierda (sin volumen ocluido)", 0.35),
        ]

        for ax, zlo, zhi, rlim, titulo, lw in ventanas:
            # Sombrear membrana solida como poligono limpio sin malla espuria
            poly_top = np.vstack([
                np.column_stack([z_line, r_line]),
                [[self.L, self.Rres], [0.0, self.Rres]]
            ])
            poly_bot = np.vstack([
                np.column_stack([z_line, -r_line]),
                [[self.L, -self.Rres], [0.0, -self.Rres]]
            ])
            ax.fill(poly_top[:, 0], poly_top[:, 1], color="#e0e0e0", alpha=0.7, label="Membrana (sólido)")
            ax.fill(poly_bot[:, 0], poly_bot[:, 1], color="#e0e0e0", alpha=0.7)

            lc_fluid = LineCollection(lines_fluid, colors="#1f77b4", linewidths=lw, alpha=0.65,
                                      label="Fluido (celdas activas)")
            ax.add_collection(lc_fluid)
            ax.plot(z_line, r_line, "r-", lw=1.5, label="Pared del Poro")
            ax.plot(z_line, -r_line, "r-", lw=1.5)
            ax.plot([0.0, 0.0], [self.Rtip, self.Rres], "r-", lw=1.5)
            ax.plot([0.0, 0.0], [-self.Rtip, -self.Rres], "r-", lw=1.5)
            ax.plot([self.L, self.L], [self.Rbase, self.Rres], "r-", lw=1.5)
            ax.plot([self.L, self.L], [-self.Rbase, -self.Rres], "r-", lw=1.5)

            ax.set_xlim(zlo, zhi)
            ax.set_ylim(-rlim, rlim)
            ax.set_title(titulo, fontsize=11)
            ax.set_xlabel("z (nm)", fontsize=10)
            ax.set_ylabel("r (nm)", fontsize=10)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(archivo_salida, dpi=300)
        plt.close(fig)
        print(f"--- Esquema completo de malla guardado en '{archivo_salida}' ---")

    def resumen(self, imprimir=True):
        z, hz = self.zf, self.dz
        raz = np.maximum(hz[1:] / hz[:-1], hz[:-1] / hz[1:])
        vol = 2 * np.pi * self.V.sum()
        vol_exact = (2 * np.pi / 2) * (self.Rres**2 * self.Lres * 2) + np.pi * _gauss_cells(
            np.linspace(0, self.L, 20001)[:-1], np.linspace(0, self.L, 20001)[1:],
            lambda zz: self.perfil(zz)**2).sum()
        out = dict(Nzc=self.Nzc, Nr=self.Nr, Nxi=self.Nxi, celdas_activas=int(self.act.sum()),
                   razon_axial_max=float(raz.max()), err_vol=float(abs(vol / vol_exact - 1)),
                   h_min=float(hz.min()), h_max=float(hz.max()))
        if imprimir:
            k0, kL = self.k0, self.kL
            print("=" * 72)
            print(f"MALLA FD: {self.Nzc} columnas z x {self.Nr} celdas r "
                  f"(Nxi={self.Nxi} en el poro/boca + {self.Nout} fuera) -> "
                  f"{out['celdas_activas']} celdas activas, {self.nunk} incognitas")
            print(f"  columnas: reservorio izq {k0}, poro {kL - k0}, reservorio der {self.Nzc - kL}")
            print(f"  celda axial en la boca: izq {hz[k0 - 1]:.4g} | der {hz[k0]:.4g} nm "
                  f"(poro) ; en z=L: {hz[kL - 1]:.4g} | {hz[kL]:.4g} nm")
            print(f"  celda radial de pared: tip {self.dxi_w * self.Rtip:.4g} nm, "
                  f"base {self.dxi_w * self.Rbase:.4g} nm")
            print(f"  h axial min/max = {out['h_min']:.4g}/{out['h_max']:.4g} nm; "
                  f"razon vecina max = {out['razon_axial_max']:.3f}")
            print(f"  error de volumen total vs analitico = {out['err_vol']:.2e}")
            print("=" * 72)
        return out