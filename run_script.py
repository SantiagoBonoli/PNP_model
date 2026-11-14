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

# ============================ parametros (nm, C/m2) ============================
E_CHARGE = 1.602176634e-19                  # C

R_TIP, R_BASE, L_H, L = 7.5, 250.0, 6.0, 12000.0    # nm (L/h del perfil "bullet")
SIGMA = -1.0 * E_CHARGE * 1e18              # carga de pared: -1 e/nm2 en C/m2
SIGMA_MEMBRANA = 0.0                        # C/m2 en las caras planas de la membrana (z=0, z=L; r>R)

V_LIST = np.linspace(-1.0, 1.0, 21)         # V = phi(base) - phi(punta)
DV_MAX = 0.05                               # paso maximo de continuacion en V (baja solo si no converge)

R_RES = 30.0 * R_BASE                       # radio de los reservorios (nm)
L_RES = R_RES                               # largo de cada reservorio (nm)
REFINAR = 2.0                               # divide TODOS los tamanos de celda de la malla

# Carga volumetrica (solo si MODO != "superficial"); densidades en e/nm3
MODO = "superficial"                        # "superficial" | "vol_z" | "vol_radial" | "vol_combinado"
CARGA = dict(
    modo_carga=MODO,
    rho_vol=0.0 * E_CHARGE * 1e27,          # C/m3 (1 e/nm3 ~ 1.66 M)
    z_vol=(0.0, 1.0),                       # nm: zona llena (vol_z, vol_combinado) o rango (vol_radial)
    S_vol=0.0,                              # nm: espesor radial desde la pared
)

# Resolucion. Las incognitas son 3*(Nz+1)*(Neta+1).
MALLA = dict(
    L=L, Rres=R_RES, Lres=L_RES, refinar=REFINAR, sigma_membrana=SIGMA_MEMBRANA,
    **CARGA,
)

ARCHIVO_DATOS = "resultado_pnp_reservorios.pkl"
FORZAR_RECALCULO = False
TOL, MAXIT, MEM = 1e-9, 300, 8              # parametros del solver (entran en la firma)

# ============================ malla ============================
P.usar_gpu(True)
m = P.Malla(R_TIP, R_BASE, L_H, **MALLA)
m.resumen()

# ============================ calculo (con cache) ============================
# La firma incluye el HASH del contenido de la malla Y del solver: si cambia cualquiera
# de los dos codigos (o cualquier parametro), se recalcula.
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
    # A NumPy antes de guardar: el .pkl no depende de CuPy ni de la GPU.
    sol = {V: tuple(np.asarray(P._cpu(a)) for a in st) for V, st in sol.items()}
    datos = {"firma": FIRMA, "I_A": I_A, "sol": sol}
    with open(ARCHIVO_DATOS, "wb") as f:
        pickle.dump(datos, f, protocol=pickle.HIGHEST_PROTOCOL)

I_nA, sol = datos["I_A"] * 1e9, datos["sol"]

# ============================ curva I-V y rectificacion ============================
def corriente_en(V):
    return I_nA[int(np.argmin(np.abs(V_LIST - V)))]
 
 
I_neg, I_pos = corriente_en(-1.0), corriente_en(1.0)
rect = abs(I_neg / I_pos) if np.isfinite(I_neg) and abs(I_pos) > 1e-14 else np.nan
print(f"Rectificacion |I(-1 V)/I(1 V)| = {rect:.6g}")

# Conservacion de la corriente a lo largo del poro (std/|media| entre caras z)
for V_ in (-1.0, 1.0):
    if V_ in sol:
        _, I_caras = S.corriente(m, tuple(P._dev(a) for a in sol[V_]), V_ / P.VT)
        print(f"  error de conservacion en V={V_:+.1f} V: {S.error_conservacion(m, I_caras):.2e}")
 
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(V_LIST, I_nA, "o-", label="PNP 2D axisimetrico con reservorios")
ax.axhline(0.0, color="k", lw=0.6)
ax.axvline(0.0, color="k", lw=0.6)
ax.set(xlabel=r"V = $\phi_{base} - \phi_{punta}$ (V)", ylabel="I (nA)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("iv_2d_reservorios.png", dpi=200)
 

def mapa(V,
         zmin_zoom=-15.0, zmax_zoom=15.0, rmax_zoom=15.0,
         zmin_poro=-R_BASE, zmax_poro=L + R_BASE, rmax_poro=1.5 * R_BASE,
         reservorios=True,
         cmax_fijo=None):
    """
    Mapas 2D de phi, c+ y c- (espejados respecto de r = 0). Una figura por voltaje.

    Filas: (1) zoom de la punta, (2) poro con una boca de reservorio a cada lado,
           (3) [si reservorios=True] dominio completo, con ambos electrodos.
    Escala de c+/c-: de 0 al maximo visible en cada panel (o hasta cmax_fijo si se
    pasa), para no saturar el contraion dentro del poro. phi: automatica.
    """
    if len(sol) == 0:
        raise ValueError("No hay soluciones disponibles en sol.")

    V_sol = min(sol.keys(), key=lambda v: abs(v - V))

    # 1. Variables (siempre a CPU)
    estado = sol[V_sol]
    phi = np.asarray(P._cpu(estado[0])) * P.VT
    cp = np.asarray(P._cpu(estado[1]))
    cm = np.asarray(P._cpu(estado[2]))
    U = np.stack((phi, cp, cm), axis=-1)

    # 2. Esquinas de celda (Nzc+1, Nr+1): z en las caras, r = promedio de m.rm entre columnas
    rm_car = np.empty((m.Nzc + 1, m.Nr + 1))
    rm_car[1:-1] = 0.5 * (m.rm[1:] + m.rm[:-1])
    rm_car[0], rm_car[-1] = m.rm[0], m.rm[-1]

    # 3. Perfil geometrico. En los reservorios se dibuja la pared exterior (Rres); el
    #    salto Rres -> perfil en z=0 / z=L traza la cara plana de la membrana.
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

    # 4. Paneles
    for fila, (zlo, zhi, rmax, nombre_fila) in enumerate(ventanas):
        sel = (m.zc >= zlo) & (m.zc <= zhi)
        if np.count_nonzero(sel) < 2:
            raise ValueError(f"Ventana {nombre_fila} con nodos insuficientes.")

        Us, acts, rhos = U[sel], m.act[sel], rho[sel]
        i0 = int(np.flatnonzero(sel)[0])
        i1 = int(np.flatnonzero(sel)[-1]) + 1
        zcar, rcar = m.zf[i0:i1 + 1], rm_car[i0:i1 + 1]

        for col, (titulo, k, es_conc) in enumerate(campos):
            ax = axs[fila, col]
            C = np.ma.masked_where(~acts, Us[:, :, k])
            valores = C.compressed()
            if valores.size == 0:
                continue

            if es_conc:
                vmin_c = 0.0
                vmax_c = float(valores.max()) if cmax_fijo is None else float(cmax_fijo)
                vmax_c = max(vmax_c, 1.0)
            else:
                vmin_c, vmax_c = float(valores.min()), float(valores.max())
                if np.isclose(vmin_c, vmax_c):
                    vmax_c = vmin_c + 1e-10

            # Espejo respecto de r = 0, con esquinas EXACTAS de celda (z: caras zf; r: caras rm
            # promediadas entre columnas vecinas), asi pcolormesh no interpola entre celdas.
            Zc = np.repeat(zcar[:, None], 2 * m.Nr + 1, axis=1)
            Rc = np.concatenate((-rcar[:, :0:-1], rcar), axis=1)
            Cfull = np.ma.concatenate((C[:, ::-1], C), axis=1)

            cf = ax.pcolormesh(Zc, Rc, Cfull, shading="flat", cmap="viridis",
                               vmin=vmin_c, vmax=vmax_c, rasterized=True)

            ax.plot(m.zc[sel], rhos, "k", lw=1.3)
            ax.plot(m.zc[sel], -rhos, "k", lw=1.3)
            ax.axvline(0.0, color="white", ls="--", lw=0.9, alpha=0.8)
            ax.axvline(m.L, color="white", ls="--", lw=0.9, alpha=0.8)

            ax.set_xlim(zlo, zhi)
            ax.set_ylim(-rmax, rmax)
            ax.set_xlabel("z - z_punta (nm)")
            if fila == 0:
                ax.set_title(titulo, fontsize=13)
            if col == 0:
                ax.set_ylabel(f"{nombre_fila}\nr (nm)")

            cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.035)
            cb.ax.tick_params(labelsize=8)

    fig.suptitle(f"Distribucion de potencial y concentraciones | V = {V_sol:+.2f} V",
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    nombre = (f"mapas_punta_V{V_sol:+.2f}_{len(ventanas)}x3.png"
              .replace("+", "p").replace("-", "m"))
    fig.savefig(nombre, dpi=200, bbox_inches="tight")
    print(f"Mapa guardado: {nombre}")
    return fig, axs


# 5. Generar mapas para cada voltaje
figuras = {}
for V in (-1.0, 0.0, 1.0):
    figuras[V] = mapa(V)

plt.show()