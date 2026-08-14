"""exp1: verify the Kepler count law + principal branch theorem for point-to-helix.

Helix X(t) = (a cos t, a sin t, b t), query P = (x,y,z), rho = sqrt(x^2+y^2), phi = atan2(y,x).
Critical equation of d^2(t):  f(t) = a*rho*sin(t-phi) + b^2*t - b*z = 0.
Kepler form: u + e sin u = M,  u = t - phi,  e = a*rho/b^2,  M = z/b - phi.
Count law (e>1, generic M):  count = 2*ceil((M+c)/(2pi)) - 2*floor((M-c)/(2pi)) - 3,
    c = arccos(-1/e) + sqrt(e^2-1), thresholds m_k = 2pi*k - c, M_k = 2pi*k + c.
e<=1: count = 1 (cylinder rho < b^2/a strictly increasing).
Principal branch theorem: the global minimizer is the root of branch k0 = round(M/2pi);
    unique unless M == pi (mod 2pi), when there are exactly two symmetric minimizers.
"""
import numpy as np
from numpy import pi, sin, cos, sqrt, arctan2, ceil, floor
from math import sin as msin
import time

rng = np.random.default_rng(20260811)
log = []

def crit(t, a, b, rho, phi, z):
    return a*rho*sin(t-phi) + b*b*t - b*z

def d2(t, a, b, rho, phi, z):
    return rho*rho + a*a + (z-b*t)**2 - 2*a*rho*cos(t-phi)

def count_formula(M, e):
    if e <= 1.0:
        return 1
    c = np.arccos(-1.0/e) + sqrt(e*e-1.0)
    if abs(((M+c)/(2*pi)) - round((M+c)/(2*pi))) < 1e-9:  # on threshold: skip
        return None
    if abs(((M-c)/(2*pi)) - round((M-c)/(2*pi))) < 1e-9:
        return None
    return 2*int(ceil((M+c)/(2*pi))) - 2*int(floor((M-c)/(2*pi))) - 3

def g(u, e, M):
    return u + e*msin(u) - M

def roots_of_g(M, e, pad=0.1):
    """Exact root set of u + e sin u = M via critical-point intervals.

    Roots lie in [M-e, M+e]. Critical points of g (g'=1+e cos u=0) are
    u = 2pi*k +- alpha, alpha = arccos(-1/e), where g takes values 2pi*k +- c
    (c = alpha + sqrt(e^2-1)). Between consecutive critical points g is
    monotone, so at most one root per interval -> complete, exact bisection.
    """
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

def newton_from(t0, a, b, rho, phi, z, iters=100):
    t = t0
    for _ in range(iters):
        f = crit(t, a, b, rho, phi, z)
        fp = b*b + a*rho*np.cos(t-phi)
        if fp == 0:
            return None
        t = t - f/fp
    return t

# ---------------- Part A: critical equation + global min via root set ----------------
def part_a():
    nA = 500
    fail = 0
    maxerr = 0.0
    for i in range(nA):
        a = rng.uniform(0.3, 3.0); b = rng.uniform(0.1, 1.5)
        x = rng.uniform(-6, 6); y = rng.uniform(-6, 6); z = rng.uniform(-15, 15)
        rho = sqrt(x*x+y*y); phi = arctan2(y, x)
        e = a*rho/(b*b)
        M = z/b - phi
        us = roots_of_g(M, e)
        ts = us + phi
        for t in ts:
            if abs(crit(t, a, b, rho, phi, z)) > 1e-9:
                fail += 1
        dmin = min(d2(t, a, b, rho, phi, z) for t in ts)
        tt = np.linspace(z/b - 30, z/b + 30, 300000)
        dd = d2(tt, a, b, rho, phi, z)
        oracle = dd.min()
        err = abs(sqrt(dmin) - sqrt(oracle))
        maxerr = max(maxerr, err)
        if err > 1e-6:
            fail += 1
    log.append(f"[A] critical eq verified: n={nA}, fail={fail}, maxerr={maxerr:.2e}")

# ---------------- Part B: count formula vs exact root set ----------------
def part_b():
    nB = 20000
    bad = 0
    for i in range(nB):
        e = rng.uniform(1.0001, 20.0)
        M = rng.uniform(-60.0, 60.0)
        cf = count_formula(M, e)
        if cf is None:
            continue
        r = roots_of_g(M, e)
        cnt = len(np.unique(np.round(r, 10)))
        if cnt != cf:
            bad += 1
            if bad <= 3:
                print(f"  MISMATCH e={e:.5f} M={M:.5f} scan={cnt} formula={cf}")
    log.append(f"[B] count formula vs exact roots: n={nB}, mismatch={bad}")

# ---------------- Part C: cylinder law (e<=1 -> unique root) ----------------
def part_c():
    nC = 2000
    bad = 0
    for i in range(nC):
        a = rng.uniform(0.3, 3.0); b = rng.uniform(0.1, 1.5)
        rho = rng.uniform(0.0, 0.999*b*b/a)
        phi = rng.uniform(-pi, pi); z = rng.uniform(-30, 30)
        e = a*rho/(b*b)
        M = z/b - phi
        r = roots_of_g(M, e)
        cnt = len(np.unique(np.round(r, 9)))
        if cnt != 1:
            bad += 1
    fpmin = 1e18
    for i in range(nC):
        e = rng.uniform(0.0, 0.999)
        us = rng.uniform(-50, 50, 200)
        fp = 1 + e*np.cos(us)
        fpmin = min(fpmin, fp.min())
    log.append(f"[C] cylinder law: n={nC}, unique-root fails={bad}, min f'(u)={fpmin:.6f} (>0 expected)")

# ---------------- Part D: thresholds = double roots ----------------
def part_d():
    nD = 300
    bad = 0
    for i in range(nD):
        e = rng.uniform(1.05, 8.0)
        c = np.arccos(-1.0/e) + sqrt(e*e-1.0)
        k = rng.integers(-10, 11)
        for Mthr, ustar in [(2*pi*k - c, -np.arccos(-1.0/e) + 2*pi*k),   # min: g'=0, g''>0
                            (2*pi*k + c,  np.arccos(-1.0/e) + 2*pi*k)]:  # max: g'=0, g''<0
            gv = ustar + e*np.sin(ustar)
            g1 = 1 + e*np.cos(ustar)
            if abs(gv - Mthr) > 1e-8 or abs(g1) > 1e-8:
                bad += 1
                print(f"  threshold fail e={e:.4f} k={k} g={gv:.6e} Mthr={Mthr:.6e} g1={g1:.6e}")
            Mlo, Mhi = Mthr - 1e-3, Mthr + 1e-3
            cl = count_formula(Mlo, e); ch = count_formula(Mhi, e)
            if cl is not None and ch is not None and abs((ch - cl) - 2) > 1e-9 and abs((ch - cl) + 2) > 1e-9:
                bad += 1
                print(f"  jump fail e={e:.4f} k={k} Mthr={Mthr:.4f} cl={cl} ch={ch}")
    log.append(f"[D] threshold structure (double root + count jump 2): n={nD*2}, fail={bad}")

# ---------------- Part E1: principal branch = global minimizer ----------------
def part_e1():
    a, b = 1.0, 0.3
    nE = 20000
    bad = 0
    for i in range(nE):
        rho = rng.uniform(0.09, 8.0)     # e in (1, 88.9)
        phi = rng.uniform(-pi, pi)
        z = rng.uniform(-12, 12)
        e = a*rho/(b*b)
        M = z/b - phi
        us = roots_of_g(M, e)
        k0 = int(round(M/(2*pi)))
        # principal root = argmin |u - 2pi*k0| (strictly smaller than all others)
        ustar = us[np.argmin(np.abs(us - 2*pi*k0))]
        dvals = [d2(u + phi, a, b, rho, phi, z) for u in us]
        dmin = min(dvals)
        if abs(d2(ustar + phi, a, b, rho, phi, z) - dmin) > 1e-9:
            bad += 1
            if bad <= 5:
                print(f"  E1 PRINCIPLE FAIL M={M:.4f} e={e:.4f}")
        nmin = sum(1 for u in us if abs(d2(u + phi, a, b, rho, phi, z) - dmin) < 1e-6)
        if nmin != 1:
            bad += 1
    log.append(f"[E1] principal branch = global min: n={nE}, fail={bad}")

# ---------------- Part E1b: tie case M = pi (mod 2pi) -> two symmetric minimizers ----------------
def part_e1b():
    a, b = 1.0, 0.3
    phi = 0.5
    bad = 0
    for e in [1.1, 2.0, 10.0, 50.0]:
        rho = e*b*b/a
        for k in range(-2, 3):
            M = pi + 2*pi*k
            z = b*(M + phi)
            us = roots_of_g(M, e)
            dvals = np.array([d2(u + phi, a, b, rho, phi, z) for u in us])
            dmin = dvals.min()
            nm = int(np.sum(np.abs(dvals - dmin) < 1e-9))
            if nm != 2:
                bad += 1
                print(f"  E1b tie fail e={e} M={M:.4f}: minimizers={nm}, all={len(us)}")
            # symmetry: the two minimizers are +/-theta around 2pi*round
            idx = np.where(np.abs(dvals - dmin) < 1e-9)[0]
            u1, u2 = us[idx[0]], us[idx[1]]
            if abs((u1 + u2) - 2*pi*round((u1+u2)/(2*pi))) > 1e-6:
                bad += 1
    log.append(f"[E1b] tie M=pi mod 2pi: two symmetric minimizers, fail={bad}")

# ---------------- Part E2: naive start u0 = M (no principal rule) ----------------
def part_e2():
    a, b = 1.0, 0.3
    nE = 20000
    nwrong = 0
    nconvg = 0
    nfail = 0
    for i in range(nE):
        rho = rng.uniform(0.09, 8.0)
        phi = rng.uniform(-pi, pi)
        z = rng.uniform(-12, 12)
        e = a*rho/(b*b)
        M = z/b - phi
        us = roots_of_g(M, e)
        dvals = [d2(u + phi, a, b, rho, phi, z) for u in us]
        dmin = min(dvals)
        # naive start: t0 = z/b (u0 = M), i.e. projection onto central line, no turn rounding
        tN = newton_from(z/b, a, b, rho, phi, z)
        if tN is None or abs(crit(tN, a, b, rho, phi, z)) > 1e-6:
            nfail += 1
            continue
        nconvg += 1
        if abs(d2(tN, a, b, rho, phi, z) - dmin) > 1e-6:
            nwrong += 1
    log.append(f"[E2] naive start u0=M: n={nE}, converge={nconvg}, wrong-min={nwrong} ({100.0*nwrong/nE:.2f}%), no-converge={nfail}")

# ---------------- Part F: count-region classification vs analytic surfaces ----------------
def part_f():
    a, b = 1.0, 0.3
    phi = 0.7
    rho_grid = np.linspace(0.02, 8.0, 200)
    z_grid = np.linspace(-20, 20, 200)
    count_map = np.zeros((len(rho_grid), len(z_grid)), dtype=int)
    for i, rho in enumerate(rho_grid):
        e = a*rho/(b*b)
        for j, z in enumerate(z_grid):
            M = z/b - phi
            cf = count_formula(M, e)
            r = roots_of_g(M, e)
            cnt = len(np.unique(np.round(r, 8)))
            count_map[i, j] = cnt
            if cf is not None and cnt != cf:
                print(f"  [F] mismatch rho={rho:.4f} z={z:.4f} scan={cnt} formula={cf}")
                return False
    # verify boundaries: at z = b*(phi + 2pi*k + c(e)) count jumps
    rho = 1.0; e = a*rho/(b*b); c = np.arccos(-1.0/e) + sqrt(e*e-1.0)
    zs = np.linspace(-30, 30, 30001)
    cnts = []
    for z in zs:
        M = z/b - phi
        cf = count_formula(M, e)
        cnts.append(cf if cf is not None else len(np.unique(np.round(roots_of_g(M, e), 8))))
    jumps = []
    for j in range(1, len(zs)):
        if cnts[j] != cnts[j-1]:
            jumps.append(0.5*(zs[j-1]+zs[j]))
    an = []
    # z_k^+ = b*(phi + 2pi*k + c): k ranges from floor((zmin/b - phi - c)/2pi) to ceil((zmax/b - phi - c)/2pi)
    k1 = int(np.floor((-30.0/b - phi - c)/(2*pi)))
    k2 = int(np.ceil((30.0/b - phi - c)/(2*pi)))
    for k in range(k1, k2+1):
        an.append(b*(phi + 2*pi*k + c))
    # z_k^- = b*(phi + 2pi*k - c): k ranges from floor((zmin/b - phi + c)/2pi) to ceil((zmax/b - phi + c)/2pi)
    k3 = int(np.floor((-30.0/b - phi + c)/(2*pi)))
    k4 = int(np.ceil((30.0/b - phi + c)/(2*pi)))
    for k in range(k3, k4+1):
        an.append(b*(phi + 2*pi*k - c))
    an = np.array(sorted(an))
    an = an[(an >= -30.0) & (an <= 30.0)]
    errs = []
    for jz in jumps:
        d = np.min(np.abs(an - jz))
        errs.append(d)
    log.append(f"[F] region classification: jumps={len(jumps)} vs analytic={len(an)}; max loc err={max(errs):.2e}")
    return True

# ---------------- Part G: conditioning near thresholds + Lipschitz value ----------------
def part_g():
    e = 10.0
    c = np.arccos(-1.0/e) + sqrt(e*e-1.0)
    Mthr = -c                       # min threshold at k=0
    ustar = -np.arccos(-1.0/e)      # double root
    g2 = sqrt(e*e-1.0)              # g''(ustar)
    cth = 2.0*sqrt(2.0/g2)          # predicted sep/sqrt(delta)
    ratios = []
    for j in range(2, 13):
        delta = 10.0**(-j)
        us = roots_of_g(Mthr + delta, e)
        near = sorted([u for u in us if abs(u - ustar) < 1.0])
        if len(near) != 2:
            log.append(f"[G] WARNING: near-threshold root count {len(near)} at delta={delta:.1e}")
            continue
        ratios.append((near[1] - near[0])/sqrt(delta))
    rel = abs(ratios[-1] - cth)/cth
    log.append(f"[G] root sep ~ C*sqrt(delta): C_num={ratios[-1]:.4f} vs C_th={cth:.4f} (delta=1e-12), rel err {rel:.2e}")
    # value Lipschitz: d(P) = min_t |X(t)-P| is 1-Lipschitz in P
    a, b = 1.0, 0.3
    bad = 0
    for i in range(2000):
        rho = rng.uniform(0.0, 8.0)
        phi = rng.uniform(-pi, pi)
        z = rng.uniform(-12, 12)
        M = z/b - phi
        e2 = a*rho/(b*b)
        us = roots_of_g(M, e2)
        d0 = min(d2(u + phi, a, b, rho, phi, z) for u in us)
        h = rng.uniform(-1e-3, 1e-3, 3)
        rho2 = sqrt(max((rho*np.cos(phi) + h[0])**2 + (rho*np.sin(phi) + h[1])**2, 0.0))
        phi2 = np.arctan2(rho*np.sin(phi) + h[1], rho*np.cos(phi) + h[0])
        z2 = z + h[2]
        M2 = z2/b - phi2
        e3 = a*rho2/(b*b)
        us2 = roots_of_g(M2, e3)
        d1 = min(d2(u + phi2, a, b, rho2, phi2, z2) for u in us2)
        if abs(sqrt(d1) - sqrt(d0)) > np.linalg.norm(h)*(1.0 + 1e-9):
            bad += 1
    log.append(f"[G] distance value is 1-Lipschitz in query: n=2000, violations={bad}")

t0 = time.time()
part_a(); part_b(); part_c(); part_d(); part_e1(); part_e1b(); part_e2(); ok = part_f(); part_g()
log.append(f"total time {time.time()-t0:.1f}s, part_f ok={ok}")
out = "\n".join(log)
print(out)
with open("exp1_kepler_count.log", "w") as f:
    f.write(out + "\n")
