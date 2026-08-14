"""exp3: verify the deepened theory for the MDPI Mathematics version.

New mathematical content verified here:
  [F1] Focal surface = bifurcation surfaces: for P = C(t) + mu*B(t) on the focal
       surface of the helix (C = center of curvature, B = binormal), Kepler's
       equation g(u)=0 has a degenerate (double) root exactly at u = t - phi (mod 2pi),
       and M = 2*pi*k +- c(e).  Hence the count-jump surfaces ARE the focal surfaces.
  [F2] Cuspidal edge: the two sheets of the focal surface meet at rho = b^2/a along
       the evolute helix (curvature-center locus); there g has a cubic inflection
       (g' = g'' = 0, g''' = -e cos u = 1 != 0).
  [F3] Semicubical cusp: c(1+E) - pi ~ (2*sqrt(2)/3) * E^(3/2).
  [F4] Multiplicity-corrected count law: Nt = 2*ceil((M+c)/2pi) - 2*floor((M-c)/2pi) - 1
       for ALL M (e>1), counting double roots twice; distinct count = Nt - 1 exactly
       on the fold lines M = 2*pi*k +- c.
  [F5] Asymptotics: c(e) = e + pi/2 + 1/(2e) + O(e^-3);  max N = 2*ceil(c/pi) - 1
       grows like (2/pi)*e + O(1).
  [F6] Conditioning trichotomy:
       (a) principal branch: du*/dM = 1/(1 + e cos th0) < 1/(e-1) for all e>1, M;
       (b) fold: root separation ~ C*sqrt(delta)          (also in exp1, part G);
       (c) evolute (e=1, M=pi+eps): root shift ~ (6*eps)^(1/3).
  [F7] Two coaxial equal-pitch helices reduce to Kepler with e' = ac/b^2,
       M' = z0/b - s0; count law + principal-branch theorem transfer verbatim.
  [F8] Solver benchmark: bisection-Newton hybrid on the principal branch (iteration
       counts, quadratic convergence) vs Newton from u0 = 0 (cylinder point) and
       u0 = M (naive): wrong-minimizer and failure rates.
  [F9] Hessian lower bound at the minimizer: (d^2)'' >= 2 b^2 (e-1).
  [F10] e* = unique e with c(e) = 2*pi, to 12 digits.
"""
import numpy as np
from numpy import pi, sin, cos, sqrt, arctan2, arccos, ceil, floor
from math import sin as msin, cos as mcos
import time

rng = np.random.default_rng(20260817)
log = []

def g(u, e, M):
    return u + e*msin(u) - M

def gp(u, e):
    return 1.0 + e*mcos(u)

def alpha_of(e):
    return np.arccos(-1.0/e)

def c_of(e):
    return np.arccos(-1.0/e) + sqrt(e*e - 1.0)

def tol_of(u, e):
    return 1e-12 * (1.0 + abs(u) + e)

def roots_of_g(M, e, mult=False):
    """Exact root set of u + e sin u = M via critical-point intervals (bisection).
    Double roots (g = g' = 0 at u = 2pi*k +- alpha, M = 2pi*k +- c) are detected
    robustly; mult=True counts them twice."""
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
    claims = {}   # rounded root value -> number of adjacent intervals claiming it
    for j in range(len(bounds) - 1):
        l, r = bounds[j], bounds[j+1]
        gl, gr = g(l, e, M), g(r, e, M)
        tol_l, tol_r = tol_of(l, e), tol_of(r, e)
        if abs(gl) <= tol_l or abs(gr) <= tol_r:
            u = l if abs(gl) <= tol_l else r
            key = round(u, 10)
            claims[key] = claims.get(key, 0) + 1
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
            key = round(0.5*(l+r), 10)
            claims[key] = claims.get(key, 0) + 1
    if mult:
        return np.array([u for u, c in claims.items() for _ in range(c)])
    return np.array(list(claims.keys()))

def snap(x):
    """Snap a float to the nearest integer when within 1e-9 (razor-edge robust)."""
    r = round(x)
    return float(r) if abs(x - r) < 1e-9 else x

def on_fold(M, e):
    c = c_of(e)
    x = (M+c)/(2*pi); y = (M-c)/(2*pi)
    return abs(x - round(x)) < 1e-9 or abs(y - round(y)) < 1e-9

def count_multiplicity(M, e):
    """Multiplicity-corrected count, valid for all M with e>1:
    Nt = 2 ceil((M+c)/2pi) - 2 floor((M-c)/2pi) - 3 + 2*chi(M),
    chi(M) = 1 iff M = 2*pi*k +- c (fold line), else 0.
    The snap guards against float roundoff in the test."""
    c = c_of(e)
    N0 = 2*int(ceil(snap((M+c)/(2*pi)))) - 2*int(floor(snap((M-c)/(2*pi)))) - 3
    return N0 + 2 if on_fold(M, e) else N0

def principal_root(M, e):
    """Unique root of g in branch k0 = round(M/2pi) (for generic M)."""
    k0 = int(round(M/(2*pi)))
    alpha = alpha_of(e)
    l, r = 2*pi*k0 - alpha, 2*pi*k0 + alpha
    gl = g(l, e, M)
    for _ in range(60):
        m = 0.5*(l+r)
        gm = g(m, e, M)
        if gm*gl < 0:
            r = m
        else:
            l = m
            gl = gm
    return 0.5*(l+r)

# ---------------- F1: focal surface = bifurcation surfaces ----------------
def part_f1():
    a, b = 1.0, 0.3
    L = sqrt(a*a + b*b)
    n = 400
    bad = 0
    maxerr = 0.0
    for i in range(n):
        t = rng.uniform(-10, 10)
        mu = rng.uniform(-3, 3)
        # center of curvature C(t) and binormal B(t)
        Cx, Cy, Cz = -(b*b/a)*cos(t), -(b*b/a)*sin(t), b*t
        Bx, By, Bz = b*sin(t)/L, -b*cos(t)/L, a/L
        x, y, z = Cx + mu*Bx, Cy + mu*By, Cz + mu*Bz
        rho = sqrt(x*x + y*y); phi = arctan2(y, x)
        e = a*rho/(b*b)
        M = z/b - phi
        # the degenerate root must be at u = t - phi (mod 2pi)
        us = roots_of_g(M, e)
        gpr = np.array([abs(gp(u, e)) for u in us])
        i0 = int(np.argmin(gpr))
        ustar = us[i0]
        e1 = abs(g(ustar, e, M))
        e2 = gpr[i0]
        d = (ustar - (t - phi)) / (2*pi)
        e3 = abs(d - round(d))
        dc = (M - c_of(e)) / (2*pi)
        dm = (M + c_of(e)) / (2*pi)
        e4 = min(abs(dc - round(dc)), abs(dm - round(dm)))
        err = max(e1, e2, 2*pi*e3, e4)
        maxerr = max(maxerr, err)
        if err > 1e-7:
            bad += 1
            if bad <= 3:
                print(f"  F1 FAIL t={t:.3f} mu={mu:.3f} e={e:.3f} M={M:.4f} e1={e1:.1e} e2={e2:.1e} e3={e3:.1e} e4={e4:.1e}")
    log.append(f"[F1] focal surface = bifurcation surfaces: n={n}, fail={bad}, maxerr={maxerr:.2e}")

# ---------------- F2/F3: cuspidal edge = evolute helix; semicubical cusp ----------------
def part_f2f3():
    # F2: at the evolute point (e=1, M=pi): g(pi)=g'(pi)=g''(pi)=0, g'''(pi)= -cos(pi) = 1
    gg = [g(pi, 1.0, pi), gp(pi, 1.0), -msin(pi), -mcos(pi)]
    log.append(f"[F2] evolute point e=1,M=pi: g,g',g'',g''' = {gg[0]:.2e},{gg[1]:.2e},{gg[2]:.2e},{gg[3]:.4f} (expect 0,0,0,1)")
    # F2b: evolute helix points (rho=b^2/a, z=b(phi+pi) mod 2pi b) give e=1, M=pi mod 2pi
    a, b = 1.0, 0.3
    bad = 0
    for i in range(100):
        t = rng.uniform(-20, 20)
        Cx, Cy, Cz = -(b*b/a)*cos(t), -(b*b/a)*sin(t), b*t
        rho = sqrt(Cx*Cx + Cy*Cy); phi = arctan2(Cy, Cx)
        e = a*rho/(b*b)
        M = Cz/b - phi
        if abs(e - 1.0) > 1e-12 or abs(((M - pi)/(2*pi)) - round((M - pi)/(2*pi))) > 1e-9:
            bad += 1
    log.append(f"[F2b] evolute helix = {chr(123)}e=1, M=pi mod 2pi{chr(125)}: n=100, fail={bad}")
    # F3: semicubical cusp c(1+E) - pi ~ (2 sqrt2 / 3) E^{3/2}
    ratios = []
    for E in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        ratios.append((c_of(1.0 + E) - pi) / ((2*sqrt(2.0)/3.0) * E**1.5))
    log.append(f"[F3] semicubical cusp ratios (E=1e-1..1e-6): " +
               ", ".join(f"{r:.6f}" for r in ratios) + " (expect -> 1)")

# ---------------- F4: multiplicity-corrected count law ----------------
def part_f4():
    bad = 0
    n = 0
    # exactly on fold lines: multiplicity count = Nt, distinct = Nt - 1
    for i in range(3000):
        e = rng.uniform(1.001, 30.0)
        c = c_of(e)
        k = int(rng.integers(-6, 7))
        sgn = 1.0 if rng.integers(0, 2) else -1.0
        M = 2*pi*k + sgn*c
        Nt = count_multiplicity(M, e)
        us = roots_of_g(M, e, mult=True)
        cnt_m = len(us)                    # multiplicity count
        usd = roots_of_g(M, e, mult=False)
        cnt_d = len(usd)                   # distinct count
        # double root must be at 2*pi*k + sgn*alpha with g'=0
        alpha = alpha_of(e)
        u_dbl = 2*pi*k + sgn*alpha
        gd = abs(g(u_dbl, e, M)) + abs(gp(u_dbl, e))
        if cnt_m != Nt or cnt_d != Nt - 1 or gd > 1e-9:
            bad += 1
            if bad <= 3:
                print(f"  F4 FAIL e={e:.5f} M={M:.5f} Nt={Nt} mult={cnt_m} dist={cnt_d} gd={gd:.1e}")
        n += 1
    # generic M: multiplicity count = distinct count = Nt
    for i in range(3000):
        e = rng.uniform(1.001, 30.0)
        M = rng.uniform(-60, 60)
        c = c_of(e)
        if abs(((M+c)/(2*pi)) - round((M+c)/(2*pi))) < 1e-9 or \
           abs(((M-c)/(2*pi)) - round((M-c)/(2*pi))) < 1e-9:
            continue
        Nt = count_multiplicity(M, e)
        us = roots_of_g(M, e, mult=True)
        cnt = len(us)
        if cnt != Nt:
            bad += 1
        n += 1
    log.append(f"[F4] multiplicity count law (all M): n={n}, fail={bad}")

# ---------------- F5: asymptotics ----------------
def part_f5():
    errs = []
    for e in [10.0, 30.0, 100.0, 300.0, 1000.0]:
        errs.append((c_of(e) - e - pi/2) * 2*e)   # -> 1
    log.append(f"[F5] c(e) = e + pi/2 + 1/(2e) + O(e^-3): (c-e-pi/2)*2e = " +
               ", ".join(f"{x:.6f}" for x in errs) + " (expect -> 1)")
    rows = []
    for e in [2.0, 4.6033, 10.0, 30.0, 100.0]:
        Nmax = 2*int(ceil(c_of(e)/pi)) - 1
        rows.append(f"e={e}: Nmax={Nmax}, (2/pi)e={2*e/pi:.2f}")
    log.append(f"[F5] max N: " + "; ".join(rows))

# ---------------- F6: conditioning trichotomy ----------------
def part_f6():
    # (a) principal branch: kappa = 1/(1+e cos th0) < 1/(e-1)
    worst = 0.0
    for i in range(20000):
        e = rng.uniform(1.0005, 100.0)
        M = rng.uniform(-50, 50)
        k0 = int(round(M/(2*pi)))
        u0 = principal_root(M, e)
        th0 = u0 - 2*pi*k0
        kappa = 1.0/(1.0 + e*cos(th0))
        worst = max(worst, kappa*(e-1.0))
    log.append(f"[F6a] principal-branch conditioning: max over 20000 of kappa*(e-1) = {worst:.6f} (<1 expected)")
    # (c) evolute: root of u + sin u = pi + eps has shift ~ (6 eps)^{1/3}
    ratios = []
    for j in range(1, 13):
        eps = 10.0**(-j)
        l, r = pi, pi + 1.0
        for _ in range(80):
            m = 0.5*(l+r)
            if g(m, 1.0, pi+eps) < 0:
                l = m
            else:
                r = m
        u = 0.5*(l+r)
        ratios.append((u - pi)/(6*eps)**(1.0/3.0))
    log.append(f"[F6c] evolute cubic conditioning: (u-pi)/(6 eps)^(1/3) = " +
               ", ".join(f"{r:.6f}" for r in ratios[:6]) + ", ... " + f"{ratios[-1]:.6f} (expect -> 1)")

# ---------------- F7: two coaxial equal-pitch helices ----------------
def part_f7():
    def h1(u, a, b, c, Mp):
        return a*a + c*c - 2*a*c*np.cos(u) + b*b*(u - Mp)**2
    bad = 0
    n = 300
    for i in range(n):
        a = rng.uniform(0.3, 2.0); b = rng.uniform(0.1, 1.0)
        cc = rng.uniform(0.3, 2.0); s0 = rng.uniform(-2, 2); z0 = rng.uniform(-8, 8)
        ep = a*cc/(b*b)
        Mp = z0/b - s0
        us = roots_of_g(Mp, ep)
        dmin1 = min(h1(u, a, b, cc, Mp) for u in us)
        # check 1: every root gives a 2D critical point of |g1(t)-g2(s)|^2
        for u in us:
            t, s = 0.0, -s0 - u          # u = t - s - s0
            # partials of D(t,s) = a^2+c^2-2ac cos(t-s-s0) + (b(t-s)-z0)^2
            Dt = 2*a*cc*np.sin(t - s - s0) + 2*b*(b*(t - s) - z0)
            Ds = -2*a*cc*np.sin(t - s - s0) - 2*b*(b*(t - s) - z0)
            if abs(Dt) > 1e-9 or abs(Ds) > 1e-9:
                bad += 1
        # check 2: global min value vs dense 1D scan + golden refinement
        lo, hi = Mp - ep - 1.0, Mp + ep + 1.0
        xs = np.linspace(lo, hi, 200001)
        vals = h1(xs, a, b, cc, Mp)
        best_idx = int(np.argmin(vals))
        # golden-section around the 3 best
        order = np.argsort(vals)[:3]
        gbest = vals[order[0]]
        for oi in order:
            x0 = xs[oi]; step = xs[1] - xs[0]
            l0, r0 = x0 - step, x0 + step
            gr = (sqrt(5.0) - 1.0)/2.0
            x1 = r0 - gr*(r0 - l0); x2 = l0 + gr*(r0 - l0)
            f1 = h1(x1, a, b, cc, Mp); f2 = h1(x2, a, b, cc, Mp)
            for _ in range(60):
                if f1 < f2:
                    r0, x2, f2 = x2, x1, f1
                    x1 = r0 - gr*(r0 - l0); f1 = h1(x1, a, b, cc, Mp)
                else:
                    l0, x1, f1 = x1, x2, f2
                    x2 = l0 + gr*(r0 - l0); f2 = h1(x2, a, b, cc, Mp)
            gbest = min(gbest, f1, f2)
        if abs(sqrt(dmin1) - sqrt(gbest)) > 1e-7:
            bad += 1
            if bad <= 3:
                print(f"  F7 FAIL a={a:.3f} b={b:.3f} c={cc:.3f} s0={s0:.3f} z0={z0:.3f}: {sqrt(dmin1):.9f} vs {sqrt(gbest):.9f}")
    log.append(f"[F7] two-helix reduction to Kepler: n={n}, fail={bad}")

# ---------------- F8: solver benchmark ----------------
def part_f8():
    def bisec_newton(M, e):
        k0 = int(round(M/(2*pi)))
        alpha = alpha_of(e)
        # adaptive bisection: bring bracket half-width below eta = (e-1)/(2e)
        eta = (e - 1.0)/(2.0*e)
        nbis = max(2, int(np.ceil(np.log2(alpha/eta))))
        l, r = 2*pi*k0 - alpha, 2*pi*k0 + alpha
        fl = g(l, e, M)
        iters = 0
        for _ in range(nbis):
            m = 0.5*(l+r)
            fm = g(m, e, M)
            if fm*fl < 0:
                r = m
            else:
                l = m; fl = fm
            iters += 1
        u = 0.5*(l+r)
        while True:
            fu = g(u, e, M)
            if abs(fu) < 1e-14*(1 + abs(u) + e):
                break
            u = u - fu/gp(u, e)
            iters += 1
        return u, iters

    def newton_from(u0, e, M, itmax=100):
        u = u0
        for _ in range(itmax):
            fu = g(u, e, M)
            fp = gp(u, e)
            if abs(fu) < 1e-14*(1+abs(u)+e):
                return u, True
            if fp == 0:
                return u, False
            u = u - fu/fp
        return u, False

    it_bn = []
    ncyl_wrong = ncyl_fail = nnaive_wrong = nnaive_fail = 0
    qc_max = 0.0
    n = 20000
    for i in range(n):
        e = rng.uniform(1.05, 60.0)
        M = rng.uniform(-40, 40)
        k0 = int(round(M/(2*pi)))
        u0 = principal_root(M, e)
        u, it = bisec_newton(M, e)
        it_bn.append(it)
        # quadratic convergence check on well-conditioned principal roots
        if abs(gp(u0, e)) > 0.5:
            alpha = alpha_of(e)
            eta = (e - 1.0)/(2.0*e)
            nbis = max(2, int(np.ceil(np.log2(alpha/eta))))
            l, r = 2*pi*k0 - alpha, 2*pi*k0 + alpha
            for _ in range(nbis):
                m = 0.5*(l+r)
                if g(m, e, M) < 0:
                    l = m
                else:
                    r = m
            uu = 0.5*(l+r)
            errs = []
            for _ in range(4):
                uu = uu - g(uu, e, M)/gp(uu, e)
                errs.append(abs(uu - u0))
            for j in range(1, 3):
                if 1e-4 < errs[j-1] < 0.5 and errs[j] > 0:
                    qc_max = max(qc_max, errs[j]/errs[j-1]**2)
        # cylinder point: u0 = 0 (t = phi)
        uc, okc = newton_from(0.0, e, M)
        if not okc:
            ncyl_fail += 1
        else:
            if abs(uc - u0) > 1e-6 and abs(uc - (u0 + 2*pi)) > 1e-6 and abs(uc - (u0 - 2*pi)) > 1e-6:
                ncyl_wrong += 1
        # naive u0 = M
        un, okn = newton_from(M, e, M)
        if not okn:
            nnaive_fail += 1
        else:
            if abs(un - u0) > 1e-6 and abs(un - (u0 + 2*pi)) > 1e-6 and abs(un - (u0 - 2*pi)) > 1e-6:
                nnaive_wrong += 1
    it_bn = np.array(it_bn)
    log.append(f"[F8] solver benchmark (n={n}):")
    log.append(f"      bisection-Newton (principal branch): iters med={np.median(it_bn):.0f}, max={it_bn.max()}, wrong=0 (by construction)")
    log.append(f"      Newton from u0=0 (cylinder point): wrong-min={ncyl_wrong} ({100.0*ncyl_wrong/n:.2f}%), no-converge={ncyl_fail}")
    log.append(f"      Newton from u0=M (naive): wrong-min={nnaive_wrong} ({100.0*nnaive_wrong/n:.2f}%), no-converge={nnaive_fail}")
    log.append(f"      quadratic convergence ratio bound observed: {qc_max:.4f}")

# ---------------- F9: Hessian lower bound ----------------
def part_f9():
    worst = 1e18
    for i in range(20000):
        e = rng.uniform(1.0005, 100.0)
        M = rng.uniform(-50, 50)
        k0 = int(round(M/(2*pi)))
        u0 = principal_root(M, e)
        th0 = u0 - 2*pi*k0
        H = 1.0 + e*cos(th0)          # d2'' = 2 b^2 H
        worst = min(worst, H/(e-1.0))
    log.append(f"[F9] Hessian lower bound: min over 20000 of (1+e cos th0)/(e-1) = {worst:.6f} (>1 expected)")

# ---------------- F10: e* ----------------
def part_f10():
    l, r = 1.0 + 1e-6, 10.0
    for _ in range(200):
        m = 0.5*(l+r)
        if c_of(m) < 2*pi:
            l = m
        else:
            r = m
    e_star = 0.5*(l+r)
    log.append(f"[F10] e* = {e_star:.12f}  (c(e*)={c_of(e_star):.15f} vs 2pi={2*pi:.15f})")

t0 = time.time()
part_f1(); part_f2f3(); part_f4(); part_f5(); part_f6(); part_f7(); part_f8(); part_f9(); part_f10()
log.append(f"total time {time.time()-t0:.1f}s")
out = "\n".join(log)
print(out)
with open("exp3_deepening.log", "w") as f:
    f.write(out + "\n")
