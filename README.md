# helix–kepler: Closest points on a circular helix and Kepler's equation

Reproduction code for the manuscript

> **Closest Points on a Circular Helix and Kepler's Equation: Exact Counting,
> Focal Surfaces, and the Principal-Branch Minimizer**  
> Peilin Luo, School of Mathematics, Northeastern University (submitted to
> *Mathematics*, MDPI)

The point-to-helix closest-point problem reduces exactly to Kepler's equation
`u + e sin u = M` with eccentricity `e = a·rho/b²` (which can exceed 1). The
paper proves the exact stationary-point count law, identifies the bifurcation
surfaces with the focal surfaces of the helix, proves the principal-branch
(global minimizer) theorem, and gives a certified solver. All statements are
verified numerically here.

## Contents

| File | What it does |
|---|---|
| `exp1_kepler_count.py` | Original verification suite (count law, cylinder law, principal branch, bifurcation crossings, fold conditioning, Lipschitz). Output: `exp1_kepler_count.log`. |
| `exp2_figures.py` | Figures 1–3 (count map in (ρ,z); 7-root structure at e=10, M=π; fold conditioning). |
| `exp3_deepening.py` | Verification of the deepened theory: focal surface = bifurcation surfaces, cuspidal edge = evolute helix, semicubical cusp, multiplicity count law for all M, asymptotics, conditioning trichotomy, Hessian/conditioning bounds, two-helix reduction, solver benchmark. Output: `exp3_deepening.log`. |
| `exp4_figures.py` | Figures 4–6 ((e,M) bifurcation diagram; 3D focal surface; conditioning trichotomy). |
| `fig1_countmap.png` … `fig6_conditioning.png` | Figures 1–6 of the manuscript. |

## Requirements and reproduction

- Python 3.9+, NumPy, matplotlib.
- Run `python exp1_kepler_count.py` and `python exp3_deepening.py`; each run
  takes 1–3 minutes on a laptop and prints/writes a log with zero failures.
- Random seeds are fixed (`rng = np.random.default_rng(20260811)` in exp1,
  `20260817` in exp3), so the reported numbers are deterministic.

## Key numbers

- Count law: 20,000 random (e,M) in e ∈ (1,10³]: **0 mismatches** vs. exact
  interval isolation.
- Multiplicity refinement (valid for all M): 6,000 cases (fold lines +
  generic): **0 mismatches**.
- Principal branch = global minimizer: 20,000 random trials: **0 failures**.
- Solver benchmark (20,000 instances, e ∈ (1.05, 60), M ∈ (−40, 40)):
  principal-branch bisection–Newton → **median 6 iterations, 0 wrong**;
  Newton from u₀ = M → **54.44% wrong minimizer**, 3.25% no convergence;
  Newton from cylinder point u₀ = 0 → **72.50% wrong**, 16.74% no convergence.
- e\* (first eccentricity with ≥ 5 stationary points) = **4.603338848752**.

## Repository

This code is published at <https://github.com/lpl8848/helix-kepler>.

## License

MIT (see `LICENSE`).

