# -*- coding: utf-8 -*-
"""
Ejemplo: curva I-V y mapas 2D para un nanoporo PET con dos reservorios.

Convencion (modulo PNP revisado): tierra en el reservorio izquierdo (lado punta),
voltaje aplicado en el derecho (lado base)  ->  V = phi(base) - phi(punta), que es
el V experimental, e I > 0 para V > 0. Por eso YA NO se invierte la curva.
"""

import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pnp2d_nanoporo_gpu_ver2 as P

if not hasattr(P, "SIGNO_I"):
    raise RuntimeError(
        "pnp2d_nanoporo_gpu es la version anterior (V a la izquierda). "
        "Reemplazalo por la version revisada; si no, la curva saldria invertida."
    )

## ---------------- parametros (nm, C/m2) ----------------
R_TIP = 7.5                  # nm; diametro de punta = 15 nm
R_BASE = 250.0               # nm; diametro de base = 500 nm
L_H = 6.0                    # L/h del perfil exponencial (bullet)
L = 12000.0                  # nm = 12 um

E_CHARGE = 1.602176634e-19   # C
sigma_e_nm2 = -1.0           # carga de la pared interna, en e/nm2
SIGMA = sigma_e_nm2 * E_CHARGE * 1e18      # C/m2

# ---- modo de carga: "superficial" | "vol_z" | "vol_radial" | "vol_combinado" ----
MODO_CARGA = "superficial"
rho_vol_e_nm3 = 0                     # e/nm3 (1 e/nm3 ~ 1.66 M); VALOR DE EJEMPLO
RHO_VOL = rho_vol_e_nm3 * E_CHARGE * 1e27  # C/m3
RHO_RAD = RHO_VOL                          # capa radial en "vol_combinado" (otra densidad si querés)
Z_VOL = (0.0, 50.0)                        # nm: zona llena (vol_z, vol_combinado) o rango (vol_radial)
S_VOL = 0.0                               # nm: espesor radial desde la pared
Z_RAD = None                               # None -> la capa radial ocupa el resto del poro

# V = phi(base) - phi(punta) (convencion experimental)
V_LIST = [0]
DV_MAX = 0.05                # 0.002 son ~500 resoluciones por rama; 0.025-0.05 suele alcanzar

R_RES = 10.0 * R_BASE
L_ACCESS = 1*R_TIP/7.5
L_ACCESS_R = L_ACCESS

MALLA = dict(
    L=L, Rres=R_RES, Laccess=L_ACCESS, Laccess_R=L_ACCESS_R,
    Nz_poro=120, Nz_res=120, Nz_access=92, Neta=256,
    dw0=0.015, dz0_access=5, dz0_res=20.0,B_mon=40.0,
    modo_carga=MODO_CARGA, rho_vol=RHO_VOL, rho_rad=RHO_RAD,
    z_vol=Z_VOL, S_vol=S_VOL, z_rad=Z_RAD,
)

FORZAR_RECALCULO = False
ARCHIVO_DATOS = "resultado_pnp_reservorios.pkl"

PARAMS = dict(
    R_TIP=R_TIP, R_BASE=R_BASE, L_H=L_H, SIGMA=float(SIGMA),
    V_LIST=tuple(float(v) for v in V_LIST), DV_MAX=DV_MAX,
    MALLA=tuple(sorted(MALLA.items())), SIGNO_I=P.SIGNO_I,
)
FIRMA = repr(PARAMS)

# ---------------- malla con reservorios ----------------
P.usar_gpu(False)
m = P.Malla(R_TIP, R_BASE, L_H, **MALLA)

from plot_malla import graficar_malla

graficar_malla(m, guardar="mallado.png")

P.verificar_malla(m, verbose=False)
assert m.simetrica, "Accesos distintos: la malla NO es especular"
m.resumen_carga(SIGMA)

print("--- Geometria ---")
print(f"Longitud total:              {m.z_total:.1f} nm")
print(f"Radio de los reservorios:    {m.Rres:.1f} nm")
print(f"Longitud de reservorio:      {m.Lres:.1f} nm (cada lado)")
print(f"Longitud de acceso:          {m.Laccess:.1f} nm (cada lado)")
print(f"Inicio del nanoporo:         z = {m.z_poro0:.1f} nm")
print(f"Fin del nanoporo:            z = {m.z_poro1:.1f} nm")
print(f"Malla:                       {(m.Nz + 1)} x {(m.N + 1)} nodos")
print(f"Incognitas:                  {m.nunk}")
print(f"Carga de pared:              {SIGMA:.6f} C/m2 ({sigma_e_nm2:.3f} e/nm2)")

# ---------------- calculo o carga ----------------
datos = None
if not FORZAR_RECALCULO and os.path.exists(ARCHIVO_DATOS):
    with open(ARCHIVO_DATOS, "rb") as f:
        cargado = pickle.load(f)
    if cargado.get("firma") == FIRMA:
        datos = cargado
        print(f"\n--- Resultados cargados de '{ARCHIVO_DATOS}' (parametros coinciden) ---")
    else:
        print(f"\n--- '{ARCHIVO_DATOS}' es de otros parametros: se recalcula ---")

if datos is None:
    print("\n--- Calculando curva I-V (resolviendo modelo PNP) ---")
    t0 = time.perf_counter()
    I_A, sol = P.curva_IV(m, V_LIST, SIGMA, verbose=True, dV_max=DV_MAX,
                          verbose_newton=False)
    print(f"-> Calculo terminado en {time.perf_counter() - t0:.1f} s")
    datos = {"firma": FIRMA, "V_LIST": V_LIST, "I_A": I_A, "sol": sol}
    with open(ARCHIVO_DATOS, "wb") as f:
        pickle.dump(datos, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"-> Guardado en '{ARCHIVO_DATOS}'.")

V_LIST, I_A, sol = datos["V_LIST"], datos["I_A"], datos["sol"]
I_nA = I_A * 1e9

# ---------------- curva I-V y rectificacion ----------------
i_menos1 = int(np.argmin(np.abs(V_LIST + 1.0)))
i_mas1 = int(np.argmin(np.abs(V_LIST - 1.0)))
if abs(I_nA[i_menos1]) > 1e-14:
    rectificacion = abs(I_nA[i_menos1] / I_nA[i_mas1])
else:
    rectificacion = np.nan
print(f"Rectificacion |I(-1 V)/I(1 V)| = {rectificacion:.6g}")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(V_LIST, I_nA, "o-", label="PNP 2D axisimetrico con reservorios")
ax.axhline(0.0, color="k", lw=0.6)
ax.axvline(0.0, color="k", lw=0.6)
ax.set_xlabel("V = $\\phi_{base} - \\phi_{punta}$ (V)")
ax.set_ylabel("I (nA)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("iv_2d_reservorios.png", dpi=200)


# ---------------- mapas 2D cerca de la punta ----------------
def mapa(V, zmin_rel=-1000.0, zmax_rel=2500.0):
    """Campos cerca de la punta; z_rel = 0 es la boca del poro (lado punta)."""
    V_sol = min(sol.keys(), key=lambda valor: abs(valor - V))
    U = sol[V_sol].reshape(m.Nz + 1, m.N + 1, 3)

    z_rel = m.z - m.z_poro0
    sel = (z_rel >= zmin_rel) & (z_rel <= zmax_rel)
    if np.count_nonzero(sel) < 2:
        raise ValueError("La ventana seleccionada no contiene suficientes nodos.")

    # Recorte una sola vez; el semiplano r<0 es el espejo del r>0
    Zs = np.broadcast_to(z_rel[sel, None], m.r[sel].shape)
    Rs = m.r[sel]
    zs, rho_s = z_rel[sel], m.rho[sel]

    # (titulo, campo, vmin, vmax); None -> escala automatica
    campos = [
        (r"$\phi$ (RT/F)", U[..., 0], None, None),
        (r"$c_+ / c_0$", U[..., 1], 0.0, 1.3),
        (r"$c_- / c_0$", U[..., 2], 0.0, 1.3),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, (titulo, campo, vmin_val, vmax_val) in zip(axs, campos):
        C = campo[sel]
        kw = dict(shading="gouraud", cmap="viridis", vmin=vmin_val, vmax=vmax_val,
                  rasterized=True)
        pc = ax.pcolormesh(Zs, Rs, C, **kw)
        ax.pcolormesh(Zs, -Rs, C, **kw)
        ax.plot(zs, rho_s, "k", lw=1.0)
        ax.plot(zs, -rho_s, "k", lw=1.0)
        ax.set_ylim(-100,100)
        ax.axvline(0.0, color="w", ls="--", lw=0.8, alpha=0.8)
        ax.set_title(f"{titulo}    V = {V_sol:+.2f} V")
        ax.set_xlabel("z - z_punta (nm)")
        fig.colorbar(pc, ax=ax)

    axs[0].set_ylabel("r (nm)")
    fig.tight_layout()
    nombre = f"mapas_punta_V{V_sol:+.2f}_reservorios.png".replace("+", "p").replace("-", "m")
    fig.savefig(nombre, dpi=200)
    print(f"Mapa guardado: {nombre}")


for V in (0):
    mapa(V, zmin_rel=-100.0, zmax_rel=100.0)

plt.show()