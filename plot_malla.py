# -*- coding: utf-8 -*-
"""
Grafica el mallado de una Malla del modelo PNP (usa m.z, m.r, m.rho, ...).
Funciona con cualquier parametro y cualquier modo_carga.

Uso (en tu script, despues de crear la malla):
    from plot_malla import graficar_malla
    m = Malla(R_TIP, R_BASE, L_H, L=L, Rres=5*R_BASE, Laccess=0.1*5*R_BASE,
              Neta=64, Nz_poro=100, modo_carga="superficial")
    graficar_malla(m, guardar="mallado.png")
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def _dibujar(ax, m, xlim, ylim, lw, titulo):
    z, r, rho = m.z, m.r, m.rho
    segs = []
    for sg in (1.0, -1.0):
        segs += [np.column_stack([z, sg * r[:, j]]) for j in range(r.shape[1])]   # r(xi=cte)
        segs += [np.column_stack([np.full(r.shape[1], z[i]), sg * r[i]])          # z = cte
                 for i in range(len(z))]
    ax.add_collection(LineCollection(segs, colors="#1f4e79", linewidths=lw))
    ax.plot(z, rho, "r", lw=1.0)
    ax.plot(z, -rho, "r", lw=1.0)
    # bordes de zonas con carga volumetrica
    for (a, b, _, _) in getattr(m, "zonas", []):
        for x in (a, b):
            ax.axvline(m.z_poro0 + x, color="g", lw=0.8, ls="--")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("z (nm)")
    ax.set_ylabel("r (nm)")


def graficar_malla(m, guardar=None, zoom_tip=(-60.0, 200.0), mostrar=True):
    Rres, Rmax = m.Rres, max(m.Rtip, m.Rbase)
    La, LaR, L = m.Laccess, m.Laccess_R, m.L
    zt = m.z[-1]
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 1, 1])

    _dibujar(fig.add_subplot(gs[0, :]), m, (m.z[0], zt), (-1.05 * Rres, 1.05 * Rres), 0.15,
             f"Dominio completo (Nz={m.Nz}, Neta={m.N}, nodos={(m.Nz+1)*(m.N+1)})")
    _dibujar(fig.add_subplot(gs[1, 0]), m, (-1.05 * La, 0.05 * La), (-1.02 * Rres, 1.02 * Rres), 0.3,
             f"Acceso izquierdo y boca (Rtip = {m.Rtip:g} nm)")
    h = 0.5 * (zoom_tip[1] - zoom_tip[0])
    _dibujar(fig.add_subplot(gs[1, 1]), m, zoom_tip, (-4.5 * m.Rtip, 4.5 * m.Rtip), 0.4,
             f"Zoom en la punta (z en [{zoom_tip[0]:g}, {zoom_tip[1]:g}] nm)")
    _dibujar(fig.add_subplot(gs[2, 0]), m, (0, L), (-1.1 * Rmax, 1.1 * Rmax), 0.2,
             "Cuerpo del poro (eje r ampliado)")
    _dibujar(fig.add_subplot(gs[2, 1]), m, (L - 0.05 * LaR, L + 1.05 * LaR), (-1.02 * Rres, 1.02 * Rres), 0.3,
             f"Boca derecha (Rbase = {m.Rbase:g} nm) y acceso derecho")

    fig.suptitle(f"Mallado PNP 2D axisimetrico - Rtip={m.Rtip:g}, Rbase={m.Rbase:g}, "
                 f"L_H={m.L_h:g}, L={m.L:g} nm, modo {m.modo_carga}", fontsize=12)
    fig.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=130)
    if mostrar:
        plt.show()
    return fig
