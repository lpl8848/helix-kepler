"""exp2: generate figures for the AML letter.

fig1: count-region map in (rho, z) with bifurcation curves z = b*(phi + 2pi*k +- c(e)).
fig2: solution structure for the 7-root case (e=10, M=pi): g(u) with roots, d^2(u) with minimizers.
fig3: conditioning log-log plot: root separation vs sqrt(delta).
"""
import numpy as np
from numpy import pi, sin, cos, sqrt, arctan2, ceil, floor
from math import sin as msin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
rc("font", family="serif", size=9)
rc("mathtext", fontset="dejavuserif")

def g(u, e, M):
    return u + e*msin(u) - M

def roots_of_g(M, e):
    M = float(M); e = float(e)
    lo, hi = M - e, M + e
    if e <= 1.0:
        l, r = lo, hi
        gl = g(l, e, M)
        for _ in range(60):
            m = 0.5*(l+r)
            gm = g(m, e, M)
            if gm*gl < 0:
                r = m
            else:
                l = m
                gl = gm
        return np.array([0.5*(l+r)])
    alpha = np.arccos(-1.0/e)
    c = alpha + sqrt(e*e - 1.0)
    cps = []
    for k in range(int(np.ceil((lo - alpha)/(2*pi))) - 1, int(np.floor((hi - alpha)/(2*pi))) + 2):
        u = 2*pi*k + alpha
        if lo <= u <= hi:
            cps.append(u)
    for k in range(int(np.ceil((lo + alpha)/(2*pi))) - 1, int(np.floor((hi + alpha)/(2*pi))) + 2):
        u = 2*pi*k - alpha
        if lo <= u <= hi:
            cps.append(u)
    cps.sort()
    bounds = [lo] + cps + [hi]
    roots = []
    for j in range(len(bounds) - 1):
        l, r = bounds[j], bounds[j+1]
        gl, gr = g(l, e, M), g(r, e, M)
        if gl == 0.0:
            roots.append(l)
            continue
        if gr == 0.0:
            roots.append(r)
            continue
        if gl*gr < 0:
            for _ in range(60):
                m = 0.5*(l+r)
                gm = g(m, e, M)
                if gm*gl < 0:
                    r = m
                else:
                    l = m
                    gl = gm
            roots.append(0.5*(l+r))
    return np.array(roots)

def count_formula(M, e):
    if e <= 1.0:
        return 1
    c = np.arccos(-1.0/e) + sqrt(e*e-1.0)
    if abs(((M+c)/(2*pi)) - round((M+c)/(2*pi))) < 1e-9:
        return None
    if abs(((M-c)/(2*pi)) - round((M-c)/(2*pi))) < 1e-9:
        return None
    return 2*int(ceil((M+c)/(2*pi))) - 2*int(floor((M-c)/(2*pi))) - 3

# ---------------- fig1: count region map ----------------
def fig1():
    a, b = 1.0, 0.3
    phi = 0.7
    zmax = 20.0
    rho_grid = np.linspace(0.02, 8.0, 400)
    z_grid = np.linspace(-zmax, zmax, 400)
    RHO, Z = np.meshgrid(rho_grid, z_grid, indexing="ij")
    E = a*RHO/(b*b)
    M = Z/b - phi
    cnt = np.zeros_like(E, dtype=int)
    cyl = E <= 1.0
    cnt[cyl] = 1
    big = ~cyl
    a_c = np.arccos(-1.0/E[big]) + np.sqrt(E[big]**2 - 1.0)
    cnt[big] = 2*np.ceil((M[big] + a_c)/(2*pi)) - 2*np.floor((M[big] - a_c)/(2*pi)) - 3
    fig, ax = plt.subplots(figsize=(3.4, 3.0), dpi=200)
    pcm = ax.pcolormesh(RHO, Z, cnt, cmap="YlGnBu", shading="auto")
    ee = np.linspace(1.0001, 8.0/0.09, 600)
    ce = np.arccos(-1.0/ee) + np.sqrt(ee*ee - 1.0)
    for k in np.arange(-2, 3):
        zplus = b*(phi + 2*pi*k + ce)
        zminus = b*(phi + 2*pi*k - ce)
        ax.plot(b*b*ee/a, zplus, "k-", lw=0.4, alpha=0.7)
        ax.plot(b*b*ee/a, zminus, "k-", lw=0.4, alpha=0.7)
    ax.axvline(b*b/a, color="r", ls="--", lw=0.9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="r", ls="--", lw=0.9,
                              label=r"cylinder boundary $\rho=b^2/a$")],
              fontsize=7, loc="upper right", framealpha=0.9)
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$z$")
    ax.set_ylim(-zmax, zmax)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$N$ (number of stationary points)")
    ax.set_title(r"Stationary-point count, $a=1,\ b=0.3,\ \varphi=0.7$", fontsize=8)
    fig.tight_layout()
    fig.savefig("fig1_countmap.png", bbox_inches="tight")
    plt.close(fig)

# ---------------- fig2: solution structure for e=10, M=pi ----------------
def fig2():
    e, M = 10.0, pi
    alpha = np.arccos(-1.0/e)
    c = alpha + sqrt(e*e - 1.0)
    us = roots_of_g(M, e)
    us = np.sort(us)
    ugrid = np.linspace(M - e - 0.3, M + e + 0.3, 4000)
    ggrid = ugrid + e*np.sin(ugrid) - M
    # d^2 (a=1, b=0.3, rho = e*b^2/a):
    a, b = 1.0, 0.3
    rho = e*b*b/a
    d2g = rho*rho + a*a + (b*(M-ugrid))**2 - 2*a*rho*np.cos(ugrid)
    d2r = np.array([rho*rho + a*a + (b*(M-u))**2 - 2*a*rho*np.cos(u) for u in us])
    dmin = d2r.min()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6), dpi=200)
    ax1.plot(ugrid, ggrid, "b-", lw=1.0)
    ax1.axhline(0, color="k", lw=0.5)
    for u in us:
        ax1.plot([u], [0], "ro", ms=3.5)
    ax1.set_xlabel(r"$u$")
    ax1.set_ylabel(r"$g(u)=u+e\sin u-M$")
    ax1.set_ylim(-16.0, 12.0)
    ax1.set_title(r"$e=10,\ M=\pi$: seven roots", fontsize=9)
    ax2.plot(ugrid, d2g, "b-", lw=1.0)
    for u, d in zip(us, d2r):
        mark = "go" if abs(d - dmin) < 1e-9 else "ro"
        ax2.plot([u], [d], mark, ms=4)
    from matplotlib.lines import Line2D
    ax2.legend(handles=[Line2D([0], [0], color="g", marker="o", ls="", ms=4,
                               label="global minimizer"),
                        Line2D([0], [0], color="r", marker="o", ls="", ms=4,
                               label="other stationary points")],
               fontsize=7, loc="upper right", framealpha=0.9)
    ax2.set_xlabel(r"$u$")
    ax2.set_ylabel(r"$d^2(u)$")
    ax2.set_ylim(0.0, 9.0)
    ax2.set_title(r"two symmetric minimizers ($M\equiv\pi\ \mathrm{mod}\ 2\pi$)", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig2_structure.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2: e={e} M=pi roots={len(us)} dmin={dmin:.6f}")

# ---------------- fig3: conditioning log-log ----------------
def fig3():
    e = 10.0
    c = np.arccos(-1.0/e) + sqrt(e*e - 1.0)
    Mthr = -c
    ustar = -np.arccos(-1.0/e)
    g2 = sqrt(e*e - 1.0)
    cth = 2.0*sqrt(2.0/g2)
    deltas = 10.0**(-np.arange(1.0, 13.0))
    seps = []
    for delta in deltas:
        us = roots_of_g(Mthr + delta, e)
        near = sorted([u for u in us if abs(u - ustar) < 1.0])
        seps.append(near[1] - near[0])
    seps = np.array(seps)
    fig, ax = plt.subplots(figsize=(3.0, 2.5), dpi=200)
    ax.loglog(deltas, seps, "o-", ms=3, lw=1.0, label=r"numerical separation")
    ax.loglog(deltas, cth*np.sqrt(deltas), "k--", lw=1.0, label=r"$C\sqrt{\delta}$")
    ax.set_xlabel(r"$\delta$ (offset from threshold)")
    ax.set_ylabel("root separation")
    ax.legend(fontsize=8)
    ax.set_title(r"Quadratic tangency, $e=10$", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig3_conditioning.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3: C_num={seps[-1]/np.sqrt(deltas[-1]):.4f} vs C_th={cth:.4f}")

fig1(); fig2(); fig3()
print("figures done")
