# -*- coding: utf-8 -*-
"""
SOLVER PNP 2D axisimetrico (volumenes finitos / diferencias finitas conservativas).

SIN NEWTON. Se resuelve con una iteracion de tipo GUMMEL acelerada por ANDERSON:

  Mapa G(phi, c+, c-):
    (1) Poisson: un paso de Boltzmann linealizado (K + diag(V (c+ + c-)/l^2)) d = R_P
        -> el operador es SPD y el termino de carga es MONOTONO (problema convexo:
        el paso es de descenso), con limitacion del paso |d| <= DMAX (en unidades de V_T);
    (2) continuidad de c+ y c-: sistemas LINEALES con flujos de Scharfetter-Gummel (matriz M);
        las derivadas cruzadas de la malla mapeada del poro se tratan por correccion diferida.
  Anderson(m) minimiza en minimos cuadrados el residuo G(x)-x sobre las ultimas m iteraciones
  (acelera el punto fijo y elimina el estancamiento tipico de Gummel).

Sistemas lineales: matrices de 5 puntos sobre las celdas ACTIVAS (las celdas de membrana
se descartan: ~2x menos filas). El patron CSR es fijo para cada malla y se construye una
sola vez. Backends (SOLVER_LINEAL):
  "cudss"  : GPU. cuDSS (nvmath-python) con analisis simbolico UNA vez por malla y solo
             refactorizacion + solve en cada paso; los datos nunca salen de la GPU.
  "sparse" : LU dispersa (SuperLU) en CPU. En modo GPU solo se transfieren los ~5 arreglos
             comprimidos (unos MB) y el resultado vuelve a la GPU.
  "thomas" : eliminacion de bloques densa (legado; muy lenta en GPU por ser secuencial).
  "auto"   : GPU -> cudss si esta disponible (si falla, cae a sparse); CPU -> sparse.
El ensamblado, los flujos, los residuos y el mixing de Anderson son operaciones
vectorizadas sobre arreglos del dispositivo (CuPy).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pnp_fd_malla as MM
from pnp_fd_malla import (_xp, _cpu, _dev, F, VT, C0, ELL2, SIGHAT, D0, D_P, D_M,
                          SIGNO_I)

DMAX = 1.0            # tope del paso de potencial por iteracion (unidades V_T)
CMIN = 1e-30
FMIN, FMAX = 0.01, 20.0   # cambio relativo maximo de c por iteracion
N_POISSON = 12        # pasos internos maximos del Poisson no lineal (convexo)


def _B2(x):
    """Bernoulli estable para SG: devuelve (B(x), B(-x))."""
    xp = _xp()
    y = xp.minimum(xp.abs(x), 700.0)
    small = y < 1e-8
    ys = xp.where(small, 1.0, y)
    Bp = xp.where(small, 1.0 - 0.5 * y, ys / xp.expm1(ys))
    Bn = Bp + y
    pos = x >= 0.0
    return xp.where(pos, Bp, Bn), xp.where(pos, Bn, Bp)


# ----------------------------- gradientes y flujos -----------------------------
def _grad(m, psi):
    """d psi/dz|_xi y d psi/dr en centros de celda (para los terminos cruzados)."""
    xp, g = _xp(), m.g
    pz = xp.empty_like(psi)
    pz[1:-1] = g["dzm"] * psi[:-2] + g["dz0"] * psi[1:-1] + g["dzp"] * psi[2:]
    pz[0] = (psi[1] - psi[0]) * g["ih0"]
    pz[-1] = (psi[-1] - psi[-2]) * g["ih1"]
    pr = (xp.take_along_axis(psi, g["JP"], 1) - xp.take_along_axis(psi, g["JM"], 1)) * g["idr"]
    return pz, pr


def _pad_z(a, fila0, fila1):
    xp = _xp()
    return xp.concatenate([fila0, a, fila1], axis=0)


def _flujos(m, c, phi, zs, D, Vhat, cruz=True, solo_cruz=False):
    """Flujos (en +z, +r) de la especie: Fz (Nzc+1, Nr), Fr (Nzc, Nr-1). SG en el
    termino diagonal + terminos cruzados centrados de la malla mapeada."""
    xp, g = _xp(), m.g
    Nr = c.shape[1]
    if solo_cruz:
        Fz = xp.zeros((c.shape[0] + 1, Nr))
        Fr = xp.zeros((c.shape[0], Nr - 1))
    else:
        uno = xp.ones((1, Nr))
        cpad = _pad_z(c, uno, uno)
        ppad = _pad_z(phi, xp.zeros((1, Nr)), xp.full((1, Nr), Vhat))
        Bp, Bm = _B2(zs * (ppad[1:] - ppad[:-1]))
        Fz = D * g["wz"] * (Bp * cpad[:-1] - Bm * cpad[1:])
        Bp, Bm = _B2(zs * (phi[:, 1:] - phi[:, :-1]))
        Fr = D * g["wr"] * (Bp * c[:, :-1] - Bm * c[:, 1:])
    if cruz:
        psi = xp.log(c) + zs * phi
        pz, pr = _grad(m, psi)
        Fr = Fr + D * 0.5 * (c[:, 1:] + c[:, :-1]) * g["crz"] * 0.5 * (pz[:, 1:] + pz[:, :-1])
        Fz[1:-1] += D * 0.5 * (c[1:] + c[:-1]) * g["crr"][1:-1] * 0.5 * (pr[1:] + pr[:-1])
    return Fz, Fr


def _flujos_phi(m, phi, Vhat, solo_cruz=False):
    """Flujo J = -grad(phi) (c = 1, D = 1) con terminos cruzados."""
    xp, g = _xp(), m.g
    Nr = phi.shape[1]
    if solo_cruz:
        Fz = xp.zeros((phi.shape[0] + 1, Nr))
        Fr = xp.zeros((phi.shape[0], Nr - 1))
    else:
        ppad = _pad_z(phi, xp.zeros((1, Nr)), xp.full((1, Nr), Vhat))
        Fz = g["wz"] * (ppad[:-1] - ppad[1:])
        Fr = g["wr"] * (phi[:, :-1] - phi[:, 1:])
    pz, pr = _grad(m, phi)
    Fr = Fr + g["crz"] * 0.5 * (pz[:, 1:] + pz[:, :-1])
    Fz[1:-1] += g["crr"][1:-1] * 0.5 * (pr[1:] + pr[:-1])
    return Fz, Fr


def _div(Fz, Fr):
    div = Fz[1:] - Fz[:-1]
    div[:, :-1] += Fr
    div[:, 1:] -= Fr
    return div


# ----------------------------- residuos -----------------------------
def residuos_poisson(m, phi, cp_, cm_, Vhat, fs=1.0, fX=1.0):
    """Residuo de Poisson por celda y su escala local (para medir el error relativo)."""
    xp, g = _xp(), m.g
    sigh, sigm = m.sigma * SIGHAT * fs, m.sigma_membrana * SIGHAT
    Fz, Fr = _flujos_phi(m, phi, Vhat)
    RP = (-_div(Fz, Fr) + sigh * g["Asig"] + sigm * g["Amem"]
          + g["V"] * (cp_ - cm_ + fX * g["Xfix"]) / ELL2)
    sP = xp.abs(Fz[1:]) + xp.abs(Fz[:-1])
    sP[:, :-1] += xp.abs(Fr)
    sP[:, 1:] += xp.abs(Fr)
    sP = sP + g["V"] * (cp_ + cm_ + 1.0) / ELL2 + 1e-300
    act = g["act"]
    return xp.where(act, RP, 0.0), xp.where(act, sP, 1.0)


# ----------------------------- solver lineal tridiagonal por bloques -----------------------------
def _thomas(d, lo, up, zl, zu, b):
    """Resuelve el sistema de 5 puntos con coef.: d (diag), lo/up (j-1, j+1), zl/zu (i-1, i+1).
    Eliminacion de bloques Nr x Nr (Thomas por columnas z). Devuelve x (Nzc, Nr)."""
    xp = _xp()
    Nz, Nr = d.shape
    X, Z = [None] * Nz, [None] * Nz
    ident = xp.eye(Nr)
    Xp = Zp = None
    for i in range(Nz):
        S = xp.diag(d[i]) + xp.diag(up[i, :-1], 1) + xp.diag(lo[i, 1:], -1)
        y = b[i]
        if i > 0:
            S = S - zl[i][:, None] * Xp
            y = y - zl[i] * Zp
        sol = xp.linalg.solve(S, xp.concatenate([ident * zu[i][None, :], y[:, None]], axis=1))
        Xp, Zp = sol[:, :Nr], sol[:, Nr]
        X[i], Z[i] = Xp, Zp
    x = xp.empty_like(b)
    x[Nz - 1] = Z[Nz - 1]
    for i in range(Nz - 2, -1, -1):
        x[i] = Z[i] - X[i] @ x[i + 1]
    return x


SOLVER_LINEAL = "auto"      # "auto" | "cudss" | "sparse" | "thomas"
CHEQUEO_CUDSS = 1e-7        # residuo relativo maximo aceptado en la 1a resolucion con cuDSS


def _splu_5pt(d, lo, up, zl, zu, b):
    """(Legado) Sistema de 5 puntos completo con LU dispersa en CPU."""
    Nz, Nr = d.shape
    lo, up = lo.copy(), up.copy()
    lo[:, 0] = 0.0
    up[:, -1] = 0.0
    A = sp.diags([d.ravel(), up.ravel()[:-1], lo.ravel()[1:], zu.ravel()[:-Nr], zl.ravel()[Nr:]],
                 [0, 1, -1, Nr, -Nr], format="csc")
    return spla.splu(A).solve(b.ravel()).reshape(Nz, Nr)


class _Sistema5:
    """Patron CSR fijo del sistema de 5 puntos restringido a las celdas activas.
    Orden de las 5 diagonales por fila: (i-1,j), (i,j-1), (i,j), (i,j+1), (i+1,j), que es
    el de columnas crecientes. Se construye una vez por malla."""

    def __init__(self, m):
        act = np.asarray(_cpu(m.g["act"]))
        Nz, Nr = act.shape
        flat = np.flatnonzero(act.ravel())
        n = flat.size
        comp = -np.ones(Nz * Nr, np.int64)
        comp[flat] = np.arange(n)
        i, j = np.divmod(flat, Nr)
        vec = np.stack([flat - Nr, flat - 1, flat, flat + 1, flat + Nr], axis=1)
        ok = np.stack([i > 0, j > 0, np.ones(n, bool), j < Nr - 1, i < Nz - 1], axis=1)
        vc = np.where(ok, comp[np.clip(vec, 0, Nz * Nr - 1)], -1)
        mask = vc >= 0
        self.shape, self.n = (Nz, Nr), n
        self.indices = vc[mask].astype(np.int32)
        self.indptr = np.concatenate([[0], np.cumsum(mask.sum(1))]).astype(np.int32)
        self.flat_d = _dev(flat)
        self.mask2d_d = _dev(mask)
        self.mask_d = _dev(mask.ravel())
        self.indices_d = _dev(self.indices)
        self.indptr_d = _dev(self.indptr)
        self.cudss = None            # estado de cuDSS (se crea al primer uso)
        self.backend = None          # backend efectivo ("cudss" / "sparse")


def _sistema(m):
    s = getattr(m, "_sis5", None)
    if s is None:
        s = m._sis5 = _Sistema5(m)
    return s


def _cudss_disponible():
    if not MM.USAR_GPU:
        return False
    try:
        import cupyx.scipy.sparse  # noqa: F401
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401
        return True
    except Exception:
        return False


def _backend(sis):
    if SOLVER_LINEAL in ("sparse", "thomas"):
        return SOLVER_LINEAL
    if sis.backend is None:                    # "auto" o "cudss" (con caida a sparse si falla)
        sis.backend = "cudss" if _cudss_disponible() else "sparse"
        print(f"--- Solver lineal: {sis.backend} ({sis.n} incognitas por sistema) ---", flush=True)
    return sis.backend


def liberar_solvers(m):
    """Libera el estado de cuDSS asociado a la malla (opcional)."""
    sis = getattr(m, "_sis5", None)
    if sis is not None and sis.cudss is not None:
        try:
            sis.cudss.free()
        except Exception:
            pass
        sis.cudss = None


def _solve_cudss(sis, data, b):
    import cupy as cpx
    import cupyx.scipy.sparse as csp
    from nvmath.sparse.advanced import DirectSolver
    A = csp.csr_matrix((data, sis.indices_d, sis.indptr_d), shape=(sis.n, sis.n))
    if sis.cudss is None:
        sis.cudss = DirectSolver(A, b)
        sis.cudss.plan()                       # reordenamiento + simbolico: UNA vez
    else:
        sis.cudss.reset_operands(a=A, b=b)     # mismo patron, nuevos valores
    sis.cudss.factorize()
    x = cpx.array(sis.cudss.solve(), copy=True)
    return A, x


def _solve_sparse(sis, data, b):
    A = sp.csr_matrix((_cpu(data), sis.indices, sis.indptr), shape=(sis.n, sis.n)).tocsc()
    try:
        # matrices estructuralmente simetricas y diagonalmente dominantes: orden MMD(A+A^T)
        # y pivoteo casi diagonal (~1.7x mas rapido que COLAMD, mismo resultado a 1e-15)
        x = spla.splu(A, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.01,
                      options=dict(SymmetricMode=True)).solve(_cpu(b))
    except RuntimeError:
        raise np.linalg.LinAlgError("LU dispersa singular")
    return _dev(x)


def _resolver_5pt(m, d, lo, up, zl, zu, b):
    """Resuelve el sistema de 5 puntos (d diag; lo/up j-1,j+1; zl/zu i-1,i+1) en las celdas
    activas. Devuelve x (Nzc, Nr) con 0 en las celdas inactivas."""
    xp = _xp()
    sis = _sistema(m)
    modo = _backend(sis)
    if modo == "thomas":
        act = m.g["act"]
        d = xp.where(act, d, 1.0)
        lo, up, zl, zu = (xp.where(act, a, 0.0) for a in (lo, up, zl, zu))
        b = xp.where(act, b, 0.0)
        sc = 1.0 / xp.maximum(xp.maximum(xp.abs(d), xp.abs(lo)), xp.maximum(xp.abs(up),
                              xp.maximum(xp.abs(zl), xp.abs(zu))))
        return _thomas(d * sc, lo * sc, up * sc, zl * sc, zu * sc, b * sc)

    vals = xp.stack([zl, lo, d, up, zu], axis=-1).reshape(-1, 5)[sis.flat_d]
    vals = xp.where(sis.mask2d_d, vals, 0.0)
    sc = 1.0 / xp.max(xp.abs(vals), axis=1)                 # equilibrado por filas
    data = (vals * sc[:, None]).ravel()[sis.mask_d]
    bc = b.ravel()[sis.flat_d] * sc

    xc = None
    if modo == "cudss":
        try:
            A, xc = _solve_cudss(sis, data, bc)
            if not getattr(sis, "_chequeado", False):          # verificacion de la 1a resolucion
                res = float(xp.linalg.norm(A.dot(xc) - bc) / max(float(xp.linalg.norm(bc)), 1e-300))
                if not res < CHEQUEO_CUDSS:
                    raise RuntimeError(f"residuo relativo {res:.2e} de cuDSS")
                sis._chequeado = True
        except Exception as e:
            print(f"!! cuDSS no disponible o inexacto ({type(e).__name__}: {e}); "
                  f"se usa LU dispersa en CPU", flush=True)
            liberar_solvers(m)
            sis.backend = "sparse"
            xc = None
    if xc is None:
        xc = _solve_sparse(sis, data, bc)
    x = xp.zeros(sis.shape[0] * sis.shape[1])
    x[sis.flat_d] = xc
    return x.reshape(sis.shape)


# ----------------------------- ensamblado -----------------------------
def _matriz_poisson(m, cp_, cm_):
    xp, g = _xp(), m.g
    w, wr = g["wz"], g["wr"]
    d = w[1:] + w[:-1]
    d[:, :-1] += wr
    d[:, 1:] += wr
    d = d + g["V"] * (cp_ + cm_) / ELL2
    lo = xp.zeros_like(d)
    up = xp.zeros_like(d)
    lo[:, 1:] = -wr
    up[:, :-1] = -wr
    zl = -w[:-1].copy()
    zu = -w[1:].copy()
    zl[0] = 0.0
    zu[-1] = 0.0
    return d, lo, up, zl, zu


def _resolver_especie(m, phi, c_old, zs, D, Vhat):
    """Continuidad lineal de una especie con SG (cruzados diferidos con c_old)."""
    xp, g = _xp(), m.g
    Nr = phi.shape[1]
    ppad = _pad_z(phi, xp.zeros((1, Nr)), xp.full((1, Nr), Vhat))
    Bp, Bm = _B2(zs * (ppad[1:] - ppad[:-1]))
    w = D * g["wz"]
    d = w[1:] * Bp[1:] + w[:-1] * Bm[:-1]
    zl = -(w * Bp)[:-1]
    zu = -(w * Bm)[1:]
    rhs = xp.zeros_like(d)
    rhs[0] += w[0] * Bp[0]
    rhs[-1] += w[-1] * Bm[-1]
    zl[0] = 0.0
    zu[-1] = 0.0
    Bpr, Bmr = _B2(zs * (phi[:, 1:] - phi[:, :-1]))
    wr = D * g["wr"]
    d[:, :-1] += wr * Bpr
    d[:, 1:] += wr * Bmr
    lo = xp.zeros_like(d)
    up = xp.zeros_like(d)
    up[:, :-1] = -wr * Bmr
    lo[:, 1:] = -wr * Bpr
    Fz, Fr = _flujos(m, c_old, phi, zs, D, Vhat, solo_cruz=True)
    rhs = rhs - _div(Fz, Fr)
    c = _resolver_5pt(m, d, lo, up, zl, zu, rhs)
    return xp.clip(c, xp.maximum(FMIN * c_old, CMIN), FMAX * c_old)   # positividad + paso acotado


# ----------------------------- mapa de Gummel -----------------------------
def _poisson_boltzmann(m, st, Vhat, fs, fX):
    """Poisson NO LINEAL con los cuasi-niveles de Fermi fijos:
         c+ = exp(u_p - phi), c- = exp(phi - u_n).
    Es la minimizacion de un funcional CONVEXO (carga monotona): cada paso resuelve el
    sistema SPD (K + diag(V (c+ + c-)/l^2)) d = R_P con tope |d| <= DMAX y retroceso
    sobre el residuo; convergencia global."""
    xp = _xp()
    phi, cp_, cm_ = st
    u_p = xp.log(cp_) + phi
    u_n = phi - xp.log(cm_)
    act = m.g["act"]
    cur = phi
    for k in range(N_POISSON):
        c1, c2 = xp.exp(u_p - cur), xp.exp(cur - u_n)
        c1, c2 = xp.where(act, c1, 1.0), xp.where(act, c2, 1.0)
        RP, sP = residuos_poisson(m, cur, c1, c2, Vhat, fs, fX)
        delta = _resolver_5pt(m, *_matriz_poisson(m, c1, c2), RP)
        dm, n0 = (float(v) for v in _cpu(xp.stack([xp.max(xp.abs(delta)),
                                                   xp.max(xp.abs(RP) / sP)])))   # 1 sincronizacion
        om = min(1.0, DMAX / max(dm, 1e-300))
        for _ in range(8):
            new = cur + om * delta
            d1, d2 = xp.exp(u_p - new), xp.exp(new - u_n)
            d1, d2 = xp.where(act, d1, 1.0), xp.where(act, d2, 1.0)
            R2, sP2 = residuos_poisson(m, new, d1, d2, Vhat, fs, fX)
            if float(xp.max(xp.abs(R2) / sP2)) < n0 or om < 1e-3:
                break
            om *= 0.5
        cur = new
        if om * dm < 1e-9:
            break
    return cur


def _gummel(m, st, Vhat, fs, fX):
    phi, cp_, cm_ = st
    phi_n = _poisson_boltzmann(m, st, Vhat, fs, fX)
    cp_n = _resolver_especie(m, phi_n, cp_, +1.0, D_P, Vhat)
    cm_n = _resolver_especie(m, phi_n, cm_, -1.0, D_M, Vhat)
    return phi_n, cp_n, cm_n


def _pack(m, st):
    xp, act = _xp(), m.g["act"]
    return xp.concatenate([st[0][act], xp.log(st[1][act]), xp.log(st[2][act])])


def _unpack(m, x):
    xp, act = _xp(), m.g["act"]
    n = int(act.sum())
    out = []
    for k in range(3):
        a = xp.zeros(act.shape) if k == 0 else xp.ones(act.shape)
        a[act] = x[k * n:(k + 1) * n] if k == 0 else xp.exp(x[k * n:(k + 1) * n])
        out.append(a)
    return tuple(out)


def anderson_gummel(m, st, Vhat, fs=1.0, fX=1.0, tol=1e-9, maxit=300, mem=8, beta=1.0,
                    verbose=True):
    """Punto fijo x = G(x) con Anderson(mem). Devuelve (estado, ok, iteraciones, ||dx||inf)."""
    xp = _xp()
    x = _pack(m, st)
    Xs, Fs = [], []
    fprev = None
    hist = []
    for it in range(maxit):
        try:
            g_ = _pack(m, _gummel(m, _unpack(m, x), Vhat, fs, fX))
        except np.linalg.LinAlgError:
            return _unpack(m, xprev if it > 0 else x), False, it, float("inf")
        f = g_ - x
        fn = float(xp.max(xp.abs(f)))
        hist.append(fn)
        if verbose:
            print(f"      it {it:3d}  |G(x)-x|inf = {fn:.3e}", flush=True)
        if not np.isfinite(fn):
            return _unpack(m, x), False, it, fn
        if fn < tol:
            return _unpack(m, g_), True, it + 1, fn
        if fprev is not None:
            Xs.append(x - xprev)
            Fs.append(f - fprev)
            if len(Xs) > mem:
                Xs.pop(0)
                Fs.pop(0)
        if fprev is not None and fn > 20.0 * float(xp.max(xp.abs(fprev))):
            Xs, Fs = [], []                              # reinicio de la memoria
        xprev, fprev = x, f
        if Xs:
            dF = xp.stack(Fs, axis=1)
            dX = xp.stack(Xs, axis=1)
            Amat = dF.T @ dF
            Amat = Amat + 1e-12 * float(xp.trace(Amat)) * xp.eye(Amat.shape[0])
            gam = xp.linalg.solve(Amat, dF.T @ f)
            xn = x + beta * f - (dX + beta * dF) @ gam
            if not bool(xp.all(xp.isfinite(xn))) or float(xp.max(xp.abs(xn - x))) > 30.0:
                Xs, Fs = [], []                          # extrapolacion descontrolada: paso simple
                xn = x + beta * f
            x = xn
        else:
            x = x + beta * f
    return _unpack(m, x), False, maxit, hist[-1]


# ----------------------------- estados y continuacion -----------------------------
def estado_inicial(m):
    xp = _xp()
    return (xp.zeros(m.act.shape), xp.ones(m.act.shape), xp.ones(m.act.shape))


def resolver(m, V_volt, sigma, st0=None, tol=1e-9, maxit=300, mem=8, verbose=True,
             fcontinuacion0=0.1):
    """Estado en V_volt (derecho; izquierdo a tierra). Sin estado previo, hace continuacion
    adaptativa en la carga (superficial + volumetrica + membrana) a V=0."""
    m.sigma = float(sigma)
    Vhat = V_volt / VT
    if st0 is None:
        st = estado_inicial(m)
        f, paso = 0.0, fcontinuacion0
        while f < 1.0 - 1e-12:
            fn = min(1.0, f + paso)
            s2, ok, it, fnrm = anderson_gummel(m, st, 0.0, fs=fn, fX=fn, tol=tol,
                                               maxit=min(maxit, 120), mem=mem)
            if verbose:
                print(f"    carga f={fn:.4f} ok={ok} it={it} |dx|={fnrm:.1e}", flush=True)
            if ok:
                st, f = s2, fn
                paso = min(0.35, paso * 1.25)
            else:
                paso *= 0.5
                if paso < 1.0 / 512:
                    return st, False
        if abs(Vhat) < 1e-14:
            return st, True
    else:
        st = tuple(a.copy() for a in st0)
    st, ok, it, fnrm = anderson_gummel(m, st, Vhat, tol=tol, maxit=maxit, mem=mem)
    if verbose:
        print(f"    V={V_volt:+.3f} V  ok={ok} it={it} |dx|={fnrm:.1e}", flush=True)
    return st, ok


# ----------------------------- corriente -----------------------------
def corriente(m, st, Vhat):
    """Corriente (A) positiva para V>0. Devuelve (promedio en el poro, I en cada cara z)."""
    phi, cp_, cm_ = st
    Fp, _ = _flujos(m, cp_, phi, +1.0, D_P, Vhat)
    Fm, _ = _flujos(m, cm_, phi, -1.0, D_M, Vhat)
    tot = _cpu((Fp - Fm).sum(axis=1))
    I = SIGNO_I * 2.0 * np.pi * F * D0 * C0 * 1e-9 * tot
    return float(I[m.k0:m.kL + 1].mean()), I


def error_conservacion(m, I):
    """Desviacion relativa de la corriente entre las caras del poro (std/|media|).
    Deberia ser ~1e-6 o menor; si no, el promedio del poro esconde un problema de malla."""
    Ip = np.asarray(I)[m.k0:m.kL + 1]
    return float(Ip.std() / max(abs(Ip.mean()), 1e-300))


def curva_IV(m, V_list, sigma, verbose=True, dV_max=0.1, tol=1e-9, maxit=300, mem=8,
             verbose_iter=True):
    """Curva I-V con continuacion en V (pasos <= dV_max). I(V) - I(0) como en el modelo previo."""
    V_list = np.asarray(V_list, float)
    st0, ok = resolver(m, 0.0, sigma, tol=tol, maxit=maxit, mem=mem, verbose=verbose_iter)
    if not ok:
        raise RuntimeError("No converge el equilibrio V=0.")
    I0, _ = corriente(m, st0, 0.0)
    I = np.full(V_list.shape, np.nan)
    sol = {0.0: st0}
    for k, V in enumerate(V_list):
        if abs(V) < 1e-14:
            I[k] = 0.0
    for signo in (+1, -1):
        idx = [k for k in np.argsort(np.abs(V_list)) if signo * V_list[k] > 1e-14]
        st, Vprev, fallo = st0, 0.0, False
        for k in idx:
            V = float(V_list[k])
            Vi, dV = Vprev, dV_max
            while abs(V - Vi) > 1e-12:
                Vn = Vi + np.sign(V - Vi) * min(dV, abs(V - Vi))
                st_new, ok = resolver(m, float(Vn), sigma, st0=st, tol=tol, maxit=maxit,
                                      mem=mem, verbose=verbose_iter)
                if ok:
                    st, Vi = st_new, Vn
                    dV = min(dV_max, dV * 1.5)
                else:
                    dV *= 0.5
                    if dV < dV_max / 64:
                        print(f"  !! No converge cerca de V={Vn:+.4f} V; se conservan los puntos previos.")
                        fallo = True
                        break
            if fallo:
                break
            Vprev = V
            Ik, _ = corriente(m, st, V / VT)
            I[k] = Ik - I0
            sol[V] = st
            if verbose:
                print(f"  V={V:+.3f} V    I={I[k] * 1e9:+.6f} nA", flush=True)
    return I, sol
