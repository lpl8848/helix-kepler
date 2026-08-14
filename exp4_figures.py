"""exp4: figures for the deepened MDPI Mathematics version.

fig4: (e,M) bifurcation diagram: fold lines M = 2*pi*k +- c(e), cusp points at
      (e=1, M=(2k+1)pi), Maxwell lines M = (2k+1)pi (e>1), e* line, count shading.
fig5: 3D: the helix, its evolute helix (curvature centers), and the focal surface
      (envelope of normal planes, swept by binormal rulings through curvature centers).
fig6: conditioning trichotomy: (a) fold: root separation ~ C sqrt(delta), e=10;
      (b) evolute: root shift ~ (6 eps)^(1/3), e=1, M=pi+eps.
"""
import numpy as np
from numpy import pi, sin, cos, sqrt, arccos, ceil, floor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rc
rc("font", family="serif", size=9)
rc("mathtext", fontset="dejavuserif")

def c_of(e):
    return np.arccos(-1.0/e) + np.sqrt(e*e - 1.0)

def count_formula(M, e):
    if e <= 1.0:
        return 1
    c = c_of(e)
    return 2*np.ceil((M+c)/(2*pi)) - 2*np.floor((M-c)/(2*pi)) - 3

def fig4():
    emax, Mmax = 8.0, 14.0
    e_grid = np.linspace(1.0001, emax, 600)
    M_grid = np.linspace(-Mmax, Mmax, 600)
    E, M = np.meshgrid(e_grid, M_grid, indexing="ij")
    cE = c_of(E)
    N = 2*np.ceil((M+cE)/(2*pi)) - 2*np.floor((M-cE)/(2*pi)) - 3
    N = N.astype(int)
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
    pcm = ax.pcolormesh(E, M, N, cmap="YlGnBu", shading="auto", vmin=1, vmax=11)
    # fold lines M = 2*pi*k +- c(e)
    for k in range(-2, 3):
        c = c_of(e_grid)
        ax.plot(e_grid, 2*pi*k + c, "k-", lw=0.8)
        ax.plot(e_grid, 2*pi*k - c, "k-", lw=0.8)
    # Maxwell lines (cut locus): M = (2k+1)pi, e > 1
    for k in range(-2, 2):
        ax.plot([1.0, emax], [(2*k+1)*pi]*2, "r--", lw=1.0)
    # cusp points at (e=1, M=(2k+1)pi)
    for k in range(-2, 2):
        ax.plot([1.0], [(2*k+1)*pi], "ko", ms=4.5, mfc="white")
    # e* line
    ax.axvline(4.60334, color="g", ls=":", lw=1.2)
    ax.text(4.72, -13.2, r"$e^{*}$", color="g", fontsize=8)
    ax.set_xlim(1.0, emax)
    ax.set_ylim(-Mmax, Mmax)
    ax.set_xlabel(r"$e$")
    ax.set_ylabel(r"$M$")
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$N$ (stationary points)")
    ax.set_title(r"Bifurcation diagram in $(e,M)$", fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="k", lw=0.8, label=r"folds $M=2\pi k\pm c(e)$ (focal surface)"),
        Line2D([0], [0], color="r", ls="--", lw=1.0, label=r"Maxwell lines $M\equiv\pi$ (cut locus)"),
        Line2D([0], [0], color="k", marker="o", mfc="white", ls="", ms=4.5, label="cusps (evolute helix)")],
        fontsize=7, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig("fig4_bifurcation.png", bbox_inches="tight")
    plt.close(fig)
    print("fig4 done")

def fig5():
    a, b = 1.0, 0.3
    L = sqrt(a*a + b*b)
    t = np.linspace(0, 4*pi, 800)
    hel = np.array([a*np.cos(t), a*np.sin(t), b*t])
    evo = np.array([-(b*b/a)*np.cos(t), -(b*b/a)*np.sin(t), b*t])
    # focal surface X(t, mu) = C(t) + mu B(t)
    tt = np.linspace(0, 2*pi, 120)
    mm = np.linspace(-1.6, 1.6, 60)
    TT, MM = np.meshgrid(tt, mm, indexing="ij")
    X = -(b*b/a)*np.cos(TT) + MM*(b/L)*np.sin(TT)
    Y = -(b*b/a)*np.sin(TT) - MM*(b/L)*np.cos(TT)
    Z = b*TT + MM*(a/L)
    fig = plt.figure(figsize=(5.2, 4.0), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.25, color="C1", linewidth=0, antialiased=True)
    ax.plot(hel[0], hel[1], hel[2], "b-", lw=1.6, label="helix")
    ax.plot(evo[0], evo[1], evo[2], "r-", lw=2.2, label="evolute helix (cuspidal edge)")
    # a few binormal rulings
    for t0 in [0.6, 1.8, 3.1, 4.4]:
        C = np.array([-(b*b/a)*np.cos(t0), -(b*b/a)*np.sin(t0), b*t0])
        B = np.array([b*np.sin(t0)/L, -b*np.cos(t0)/L, a/L])
        s = np.linspace(-1.6, 1.6, 20)
        pts = C[None, :] + s[:, None]*B[None, :]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "gray", lw=0.7, alpha=0.8)
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_zlim(0, 4.4)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Helix, evolute helix, and focal surface (normal-plane envelope)", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig5_focal3d.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("fig5 done")

def fig6():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.6), dpi=200)
    # (a) fold: e=10, M = -c + delta, roots near u = -alpha
    e = 10.0
    c = c_of(e)
    alpha = np.arccos(-1.0/e)
    Mthr = -c
    ustar = -alpha
    g2 = sqrt(e*e - 1.0)
    Cth = 2.0*sqrt(2.0/g2)
    deltas = 10.0**(-np.arange(1.0, 13.0))
    seps = []
    for delta in deltas:
        M = Mthr + delta
        # isolate roots near ustar by bisection on the two monotone pieces
        # critical points near ustar: ustar - 2pi, ustar, ustar + 2pi
        lo, hi = ustar - 2*pi, ustar
        l, r = lo, hi
        gl = (l + e*np.sin(l) - M)
        for _ in range(80):
            m = 0.5*(l+r)
            gm = m + e*np.sin(m) - M
            if gm*gl < 0:
                r = m
            else:
                l = m; gl = gm
        r1 = 0.5*(l+r)
        lo, hi = ustar, ustar + 2*pi
        l, r = lo, hi
        gl = (l + e*np.sin(l) - M)
        for _ in range(80):
            m = 0.5*(l+r)
            gm = m + e*np.sin(m) - M
            if gm*gl < 0:
                r = m
            else:
                l = m; gl = gm
        r2 = 0.5*(l+r)
        seps.append(r2 - r1)
    seps = np.array(seps)
    ax1.loglog(deltas, seps, "o-", ms=3, lw=1.0, label="numerical")
    ax1.loglog(deltas, Cth*np.sqrt(deltas), "k--", lw=1.0, label=r"$C\sqrt{\delta}$")
    ax1.set_xlabel(r"$\delta$"); ax1.set_ylabel("root separation")
    ax1.legend(fontsize=8)
    ax1.set_title(r"(a) fold: $O(\sqrt{\delta})$, $e=10$", fontsize=9)
    # (b) evolute: e=1, M = pi + eps, root shift
    epsilons = 10.0**(-np.arange(1.0, 13.0))
    shifts = []
    for eps in epsilons:
        l, r = pi, pi + 1.0
        for _ in range(80):
            m = 0.5*(l+r)
            if m + np.sin(m) < pi + eps:
                l = m
            else:
                r = m
        u = 0.5*(l+r)
        shifts.append(u - pi)
    shifts = np.array(shifts)
    ax2.loglog(epsilons, shifts, "s-", ms=3, lw=1.0, label="numerical")
    ax2.loglog(epsilons, (6*epsilons)**(1.0/3.0), "k--", lw=1.0, label=r"$(6\varepsilon)^{1/3}$")
    ax2.set_xlabel(r"$\varepsilon$"); ax2.set_ylabel(r"$u-\pi$")
    ax2.legend(fontsize=8)
    ax2.set_title(r"(b) evolute: $O(\varepsilon^{1/3})$, $e=1$", fontsize=9)
    fig.tight_layout()
    fig.savefig("fig6_conditioning.png", bbox_inches="tight")
    plt.close(fig)
    print("fig6 done")

fig4(); fig5(); fig6()
print("figures done")
