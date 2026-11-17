# -*- coding: utf-8 -*-
"""
Curva I-V y mapas 2D de un nanoporo PET con dos reservorios (PNP axisimetrico).
 
Convencion: tierra en el reservorio izquierdo (lado punta), voltaje aplicado en el
derecho (lado base): V = phi(base) - phi(punta), I > 0 para V > 0.
"""
 
import os
import time
import pickle
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import pnp_fd_malla as P          # malla FD axisimetrica (poro + reservorios)
import pnp_fd_solver as S         # solver Gummel + Anderson
from matplotlib.ticker import MaxNLocator

# ============================ parametros (nm, C/m2) ============================
E_CHARGE = 1.602176634e-19                  # C

R_TIP, R_BASE, L_H, L = 7.5, 250.0, 6.0, 12000.0    # nm (L/h del perfil "bullet")
SIGMA = -1.0 * E_CHARGE * 1e18              # carga de pared: -1 e/nm2 en C/m2
SIGMA_MEMBRANA = 0.0                        # C/m2 en las caras planas de la membrana (z=0, z=L; r>R)

V_LIST = np.linspace(-1.0, 1.0, 21)         # V = phi(base) - phi(punta)
DV_MAX = 0.05                               # paso maximo de continuacion en V

R_RES = 5000                       # radio de los reservorios (nm)
L_RES = R_RES                               # largo de cada reservorio (nm)
REFINAR = 3.0                               # divide TODOS los tamanos de celda de la malla

MODO = "superficial"                        # "superficial" | "vol_z" | "vol_radial" | "vol_combinado"
CARGA = dict(
    modo_carga=MODO,
    rho_vol=0.0 * E_CHARGE * 1e27,          # C/m3
    z_vol=(0.0, 1.0),                       # nm
    S_vol=0.0,                              # nm
)

MALLA = dict(
    L=L, Rres=R_RES, Lres=L_RES, refinar=REFINAR, sigma_membrana=SIGMA_MEMBRANA,
    **CARGA,
)

ARCHIVO_DATOS = "resultado_pnp_reservorios.pkl"
ARCHIVO_IV = "curva_IV.csv" 
FORZAR_RECALCULO = False
TOL, MAXIT, MEM = 1e-9, 500, 8

# ============================ malla y exportacion ============================
P.usar_gpu(True)
m = P.Malla(R_TIP, R_BASE, L_H, **MALLA)
m.resumen()

# Exportar el mallado de manera limpia sin requerir esquema gráfico
m.exportar_mallado("mallado_nanoporo.npz")
m.visualizar_mallado("malla_esquema.png")

# ============================ calculo (con cache) ============================
def _hash(mod):
    with open(mod.__file__, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

FIRMA = repr(dict(
    R_TIP=R_TIP, R_BASE=R_BASE, L_H=L_H, SIGMA=float(SIGMA),
    V_LIST=tuple(float(v) for v in V_LIST), DV_MAX=DV_MAX,
    MALLA=sorted(MALLA.items()), SIGNO_I=P.SIGNO_I,
    TOL=TOL, MAXIT=MAXIT, MEM=MEM,
    malla_py=_hash(P), solver_py=_hash(S),
))

def cargar_cache():
    if FORZAR_RECALCULO or not os.path.exists(ARCHIVO_DATOS):
        return None
    with open(ARCHIVO_DATOS, "rb") as f:
        datos = pickle.load(f)
    if datos.get("firma") != FIRMA:
        print(f"--- '{ARCHIVO_DATOS}' es de otros parametros: se recalcula ---")
        return None
    print(f"--- Resultados cargados de '{ARCHIVO_DATOS}' ---")
    return datos

datos = cargar_cache()
if datos is None:
    print("--- Calculando curva I-V ---")
    t0 = time.perf_counter()
    I_A, sol = S.curva_IV(m, V_LIST, SIGMA, dV_max=DV_MAX, tol=TOL, maxit=MAXIT, mem=MEM)
    print(f"-> Calculo terminado en {time.perf_counter() - t0:.1f} s")
    sol = {V: tuple(np.asarray(P._cpu(a)) for a in st) for V, st in sol.items()}
    datos = {"firma": FIRMA, "I_A": I_A, "sol": sol}
    with open(ARCHIVO_DATOS, "wb") as f:
        pickle.dump(datos, f, protocol=pickle.HIGHEST_PROTOCOL)

I_nA, sol = datos["I_A"] * 1e9, datos["sol"]

# ============================ curva I-V y rectificacion ============================
def corriente_en(V):
    return I_nA[int(np.argmin(np.abs(V_LIST - V)))]
 
np.savetxt(ARCHIVO_IV, np.column_stack((V_LIST, I_nA)), delimiter=",",
           fmt=("%.4f", "%.9e"), header="V (V),I (nA)", comments="")
print(f"Tabla V-I guardada: {ARCHIVO_IV}")
print("      V (V)      I (nA)")
for v_, i_ in zip(V_LIST, I_nA):
    print(f"   {v_:+8.3f}  {i_:+12.5f}")

I_neg, I_pos = corriente_en(-1.0), corriente_en(1.0)
rect = abs(I_neg / I_pos) if np.isfinite(I_neg) and abs(I_pos) > 1e-14 else np.nan
print(f"Rectificacion |I(-1 V)/I(1 V)| = {rect:.6g}")

for V_ in (-1.0, 1.0):
    if V_ in sol:
        _, I_caras = S.corriente(m, tuple(P._dev(a) for a in sol[V_]), V_ / P.VT)
        print(f"  error de conservacion en V={V_:+.1f} V: {S.error_conservacion(m, I_caras):.2e}")
 
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(V_LIST, I_nA, "o-", label="PNP 2D axisimetrico con reservorios")
ax.axhline(0.0, color="k", lw=0.6)
ax.axvline(0.0, color="k", lw=0.6)
ax.set(xlabel=r"$\Delta V$ (V)", ylabel="I (nA)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("iv_2d_reservorios.png", dpi=200)

def mapa(V,
         zmin_zoom=-2.5, zmax_zoom=2.5, rmax_zoom=6.0,
         zmin_poro=-R_BASE, zmax_poro=L + R_BASE, rmax_poro=1.5 * R_BASE,
         reservorios=True,
         cmax_fijo=1.5,
         paso_cbar_c=0.25,
         levels=100):
    if len(sol) == 0:
        raise ValueError("No hay soluciones disponibles en sol.")

    V_sol = min(sol.keys(), key=lambda v: abs(v - V))

    estado = sol[V_sol]
    phi = np.asarray(P._cpu(estado[0])) * P.VT
    cp = np.asarray(P._cpu(estado[1]))
    cm = np.asarray(P._cpu(estado[2]))
    U = np.stack((phi, cp, cm), axis=-1)

    Z = np.broadcast_to(m.zc[:, None], (m.Nzc, m.Nr))
    R = m.rc
    rho = np.where(m.zc < 0.0, m.Rres,
                   np.where(m.zc > m.L, m.Rres, m.perfil(m.zc)))

    campos = [(r"$\phi$ (V)", 0, False),
              (r"$c_+ / c_0$", 1, True),
              (r"$c_- / c_0$", 2, True)]

    ventanas = [
        (zmin_zoom, zmax_zoom, rmax_zoom, "Zoom de la punta"),
        (zmin_poro, zmax_poro, rmax_poro, "Poro"),
    ]
    if reservorios:
        ventanas.append((float(m.zf[0]), float(m.zf[-1]), m.Rres, "Dominio completo"))

    fig, axs = plt.subplots(len(ventanas), 3, figsize=(17, 4.5 * len(ventanas)), sharey="row")

    for fila, (zlo, zhi, rmax, nombre_fila) in enumerate(ventanas):
        sel = (m.zc >= zlo) & (m.zc <= zhi)
        if np.count_nonzero(sel) < 2:
            raise ValueError(f"Ventana {nombre_fila} con nodos insuficientes.")

        Zs, Rs, Us = Z[sel, :], R[sel, :], U[sel, :, :]
        acts, rhos = m.act[sel, :], rho[sel]

        for col, (titulo, k, es_conc) in enumerate(campos):
            ax = axs[fila, col]
            C = np.ma.masked_where(~acts, Us[:, :, k])
            valores = C.compressed()
            if valores.size == 0:
                continue

            if titulo == r"$\phi$ (V)":
                vmin_c = 0.0 if V_sol > 0 else -0.5
                vmax_c = 0.5 if V_sol > 0 else 0.0
                extend = "both"
            else:
                if es_conc:
                    vmin_c = 0.0
                    vmax_c = float(valores.max()) if cmax_fijo is None else float(cmax_fijo)
                    vmax_c = max(vmax_c, paso_cbar_c)
                    extend = "max"
                else:
                    vmin_c, vmax_c = float(valores.min()), float(valores.max())
                    if np.isclose(vmin_c, vmax_c):
                        vmax_c = vmin_c + 1e-10
                    extend = "neither"

            niveles = np.linspace(vmin_c, vmax_c, levels)

            Zfull = np.concatenate((Zs[:, ::-1], Zs), axis=1)
            Rfull = np.concatenate((-Rs[:, ::-1], Rs), axis=1)
            Cfull = np.ma.concatenate((C[:, ::-1], C), axis=1)

            cf = ax.contourf(Zfull, Rfull, Cfull, levels=niveles,
                             cmap="magma", extend=extend)

            ax.plot(m.zc[sel], rhos, "k", lw=1.3)
            ax.plot(m.zc[sel], -rhos, "k", lw=1.3)
            ax.axvline(0.0, color="white", ls="--", lw=0.9, alpha=0.8)
            ax.axvline(m.L, color="white", ls="--", lw=0.9, alpha=0.8)

            ax.set_xlim(zlo, zhi)
            ax.set_ylim(-rmax, rmax)
            ax.set_xlabel("z (nm)")
            if fila == 0:
                ax.set_title(titulo, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{nombre_fila}\nr (nm)")

            cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
            if es_conc:
                cb.set_ticks(np.arange(0.0, vmax_c + 1e-9, paso_cbar_c))
                cb.ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
            else:
                cb.locator = MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10])
                cb.update_ticks()
            cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"Distribucion de potencial y concentraciones | V = {V_sol:+.2f} V",
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    nombre = (f"mapas_punta_V{V_sol:+.2f}_{len(ventanas)}x3.png"
              .replace("+", "p").replace("-", "m"))
    fig.savefig(nombre, dpi=300, bbox_inches="tight")
    print(f"Mapa guardado: {nombre}")
    return fig, axs

figuras = {}
for V in (-0.5, 0.0, 0.5):
    figuras[V] = mapa(V)

plt.show()
