# -*- coding: utf-8 -*-
"""
Curva I-V y mapas 2D de un nanoporo PET con dos reservorios (PNP axisimetrico).
 
Convencion: tierra en el reservorio izquierdo (lado punta), voltaje aplicado en el
derecho (lado base): V = phi(base) - phi(punta), I > 0 para V > 0.
"""
 
import os
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pnp_fd_malla as P          # modulo PNP (pnp_nanoporo_malla.py renombrado)
import pnp_fd_solver as S          # modulo PNP (pnp_nanoporo_malla.py renombrado)
  
# ============================ parametros (nm, C/m2) ============================
E_CHARGE = 1.602176634e-19                  # C
 
R_TIP, R_BASE, L_H, L = 7.5, 250.0, 6.0, 12000.0    # nm (L/h del perfil "bullet")
SIGMA = -1.0 * E_CHARGE * 1e18              # carga de pared: -1 e/nm2 en C/m2
 
V_LIST = np.linspace(-1.0, 1.0, 21)         # V = phi(base) - phi(punta)
DV_MAX = 0.05                               # paso maximo de continuacion en V
 
R_RES = 10.0 * R_BASE
L_ACCESS = 0.1          # boca geometrica casi en escalon
L_ACCESS_FINAL = 50                # hasta aqui crece el mallado fino del acceso
 
# Carga volumetrica (solo si MODO != "superficial"); densidades en e/nm3
MODO = "superficial"                        # "superficial" | "vol_z" | "vol_radial" | "vol_combinado"
CARGA = dict(
    modo_carga=MODO,
    rho_vol=0.0 * E_CHARGE * 1e27,          # C/m3 (1 e/nm3 ~ 1.66 M)
    z_vol=(0.0, 1.0),                      # nm: zona llena (vol_z, vol_combinado) o rango (vol_radial)
    S_vol=0.0,                             # nm: espesor radial desde la pared
)
 
# Resolucion. Las incognitas son 3*(Nz+1)*(Neta+1): la LU domina el costo, asi que
# conviene subir primero Nz_access y Nz_poro y recien despues Neta.
MALLA = dict(
    L=L, Rres=R_RES,
    **CARGA,
)
 
ARCHIVO_DATOS = "resultado_pnp_reservorios.pkl"
FORZAR_RECALCULO = False
MOSTRAR_ESQUEMA_MALLA = True
 
# ============================ malla ============================
P.usar_gpu(True)
m = P.Malla(R_TIP, R_BASE, L_H, **MALLA)
m.resumen()

# ============================ calculo (con cache) ============================
# La firma incluye la fecha del modulo del solver: si cambia el codigo, se recalcula.
FIRMA = repr(dict(
    R_TIP=R_TIP, R_BASE=R_BASE, L_H=L_H, SIGMA=float(SIGMA),
    V_LIST=tuple(float(v) for v in V_LIST), DV_MAX=DV_MAX,
    MALLA=sorted(MALLA.items()), SIGNO_I=P.SIGNO_I,
    modulo=os.path.getmtime(P.__file__),
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
    I_A, sol = S.curva_IV(m, V_LIST, SIGMA, dV_max=DV_MAX)
    print(f"-> Calculo terminado en {time.perf_counter() - t0:.1f} s")
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
 
fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(V_LIST, I_nA, "o-", label="PNP 2D axisimetrico con reservorios")
ax.axhline(0.0, color="k", lw=0.6)
ax.axvline(0.0, color="k", lw=0.6)
ax.set(xlabel=r"V = $\phi_{base} - \phi_{punta}$ (V)", ylabel="I (nA)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("iv_2d_reservorios.png", dpi=200)
 
 
# ============================ mapas 2D cerca de la punta ============================

# ============================ mapas 2D cerca de la punta ============================

def mapa(V, zmin=-100.0, zmax=100.0, r_max=100.0):
    """
    Mapas 2D de potencial phi, c+ y c- cerca de la punta.

    z = 0: boca del poro, lado punta.
    r > 0: semiplano superior.
    r < 0: imagen especular.

    Utiliza las coordenadas fisicas de los centros de celda
    de la malla actual: m.zc y m.rc.
    """

    # -------------------------------------------------
    # 1. Seleccionar la solucion mas cercana al voltaje
    # -------------------------------------------------
    if len(sol) == 0:
        raise ValueError("No hay soluciones disponibles en sol.")

    V_sol = min(sol.keys(), key=lambda v: abs(v - V))

    print(V)
    # -------------------------------------------------
    # 2. Recuperar las tres variables de estado
    # -------------------------------------------------
    # sol[V_sol] = (phi, c+, c-)
    # Cada variable tiene dimensiones (Nzc, Nr).

    estado = sol[V_sol]

    phi = np.asarray(P._cpu(estado[0]))
    cp = np.asarray(P._cpu(estado[1]))
    cm = np.asarray(P._cpu(estado[2]))

    U = np.stack((phi, cp, cm), axis=-1)

    # -------------------------------------------------
    # 3. Coordenadas fisicas de los centros de celda
    # -------------------------------------------------
    # zc: (Nzc,)
    # rc: (Nzc, Nr)

    Z = np.broadcast_to(
        m.zc[:, None],
        (m.Nzc, m.Nr)
    )

    R = m.rc

    # -------------------------------------------------
    # 4. Seleccionar ventana espacial
    # -------------------------------------------------
    sel = (m.zc >= zmin) & (m.zc <= zmax)

    if np.count_nonzero(sel) < 2:
        raise ValueError(
            "La ventana seleccionada no contiene suficientes nodos."
        )

    Z = Z[sel, :]
    R = R[sel, :]
    U = U[sel, :, :]

    # -------------------------------------------------
    # 5. Enmascarar las celdas solidas del nanoporo
    # -------------------------------------------------
    act = m.act[sel, :]

    # -------------------------------------------------
    # 6. Variables a representar
    # -------------------------------------------------
    campos = [
        (r"$\phi$ (RT/F)", 0, None, None),
        (r"$c_+ / c_0$", 1, 0.0, 1.3),
        (r"$c_- / c_0$", 2, 0.0, 1.3),
    ]

    fig, axs = plt.subplots(
        1, 3,
        figsize=(15, 4.5),
        sharey=True
    )

    # -------------------------------------------------
    # 7. Representacion de los campos
    # -------------------------------------------------
    for ax, (titulo, k, vmin, vmax) in zip(axs, campos):

        C = np.ma.masked_where(
            ~act,
            U[:, :, k]
        )

        kw = dict(
            shading="auto",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            rasterized=True
        )

        # Semiplano superior
        pc = ax.pcolormesh(
            Z,
            R,
            C,
            **kw
        )

        # Semiplano inferior: espejo respecto al eje z
        ax.pcolormesh(
            Z[:, ::-1],
            -R[:, ::-1],
            C[:, ::-1],
            **kw
        )

        # -------------------------------------------------
        # 8. Perfil geometrico del nanoporo
        # -------------------------------------------------
        # En los reservorios el radio de la boca es constante.
        # Dentro del poro se utiliza el perfil real rho(z).

        rho = np.where(
            m.zc < 0.0,
            m.Rres,
            np.where(
                m.zc > m.L,
                m.Rres,
                m.perfil(m.zc)
            )
        )

        rho = rho[sel]

        ax.plot(Z[:, 0], rho, "k", lw=1.0)
        ax.plot(Z[:, 0], -rho, "k", lw=1.0)

        # Boca del poro
        ax.axvline(
            0.0,
            color="w",
            ls="--",
            lw=0.8,
            alpha=0.8
        )

        ax.set(
            ylim=(-r_max, r_max),
            xlim=(zmin, zmax),
            xlabel="z - z_punta (nm)",
            title=f"{titulo}    V = {V_sol:+.2f} V"
        )

        fig.colorbar(pc, ax=ax)

    axs[0].set_ylabel("r (nm)")

    fig.tight_layout()

    # -------------------------------------------------
    # 9. Guardar figura
    # -------------------------------------------------
    nombre = (
        f"mapas_punta_V{V_sol:+.2f}_reservorios.png"
        .replace("+", "p")
        .replace("-", "m")
    )

    fig.savefig(
        nombre,
        dpi=200,
        bbox_inches="tight"
    )

    print(f"Mapa guardado: {nombre}")

    return fig, axs

 
for V in (-1.0, 0, 1.0):
    mapa(V)
 
plt.show()