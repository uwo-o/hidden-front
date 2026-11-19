# hidden-front

**Ratio objectives hide the Pareto front from weighted-sum decomposition.**

This repository contains the code, data and manuscripts of a study on how the choice of objectives
decides whether a weighted sum (a linear figure of merit) can reach a Pareto front. The case study is a
finite-time quantum Otto engine whose optimal cycles saturate a thermodynamic uncertainty relation.

## Main result

Many engineering and physics problems mix a **ratio** objective (efficiency, yield) with an
**extensive** output (power, throughput) and a cost that is proportional to the output at fixed ratio.

- Written with the ratio, such a front is a **saddle surface**: no interior point can be reached by any
  weighted sum, whatever the weights (Theorem 1).
- Rewritten with an extensive cost (for example, entropy production instead of efficiency), the
  **same solutions** form a perspective function, which is convex exactly when a one-dimensional profile
  is convex (Theorem 2). The convex case is flat along rays, so weighted-sum optima are not unique and
  weighted-sum decomposition crowds its solutions (Corollary 1).

For the single-qubit Otto engine:

| | Efficiency formulation $(P,\eta,\Delta P)$ | Entropy formulation $(P,\Delta P,\Sigma)$ |
|---|---|---|
| Front unreachable by any weighted sum | 91% | 4% |
| HV of the reference front / ceiling of any weighted sum | 1.0810 / **0.9950** | 1.2114 / 1.2113 |
| MOEA/D with a weighted sum: reachability + distribution loss | **0.086** + 0.028 | 0.0001 + **0.089** |
| Median loss of the best cycle a weighted-sum designer can reach | 14% | 0% |

Six of the seven multi-objective evolutionary algorithms tested exceed the hypervolume ceiling of any
weighted-sum method in the efficiency formulation. A 17-variable piecewise-control variant that breaks
the structure has a nearly convex front.

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/uwo-o/hidden-front.git
cd hidden-front
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
pytest
```

## Package

`hidden_front/` is a small library:

| Module | Contents |
|---|---|
| `otto` | Single-qubit Otto engine: limit cycle, power, efficiency, entropy production and exact power fluctuations from the tilted transfer matrix; Monte Carlo validation |
| `piecewise` | Quantum-dot engine of Erdman et al. (2023) with N-step piecewise-constant control and exact statistics |
| `problems` | pymoo problems for both engines in the ratio (`objectives="eta"`) and extensive (`objectives="sigma"`) formulations |
| `moead` | MOEA/D (normalized Tchebycheff), MOEA/D-AWA (Qi et al. 2014) and MOEA/D with a linear weighted sum |
| `support` | Linear-programming support test: is a Pareto point reachable by some weighted sum? |

```python
import numpy as np
from hidden_front.otto import cycle_statistics
from hidden_front.problems import OttoProblem
from hidden_front.moead import MOEAD
from hidden_front.support import support_gap

s = cycle_statistics(omega_h=2.04, omega_c=1.43, tau_h=0.675, tau_c=0.675)
print(s["P"], s["eta"], s["dP"], s["Sigma"])        # power, efficiency, fluctuations, entropy production

problem = OttoProblem(objectives="eta")              # minimize (-P, -eta, dP)
result = MOEAD(problem, ws=True, awa=True, wag=10, max_gen=400, seed=0).run()
F = result.result_F
gaps = support_gap(F, F)                             # 0 = supported, > 0 = unreachable by any weighted sum
```

## Reproducing the results

All scripts are run from the repository root. Times are for a 24-core desktop; the support tests are
single-threaded.

| Step | Command | Output | Time |
|---|---|---|---|
| Validate MOEA/D on DTLZ2/DTLZ7 | `python scripts/run_benchmarks.py --seeds 10 --gens 300` | `results/benchmarks.json` | ~5 min |
| Baseline Otto runs and dense reference front | `python scripts/run_otto_baseline.py --objectives eta` and `--objectives sigma` | `results/otto*_runs.json`, `otto*_reference.npz` | ~5 min each |
| Baseline analysis and figures 1-9 | `python scripts/analyze_otto_baseline.py`, `python scripts/compare_formulations.py` | `results/otto_analysis.json`, `formulation_comparison.json` | ~5 min |
| Main experiment, Otto | `python scripts/run_experiment.py --model otto --seeds 30 --gens 400 --jobs 24` | `results/ot_*_runs.json` | ~4 min |
| Main experiment, piecewise | `python scripts/run_experiment.py --model piecewise --seeds 30 --gens 3000 --jobs 24` then `python scripts/refine_reference.py` | `results/pw_*_runs.json`, `pw_*_long.json` | ~2.5 h |
| Analysis of the main experiment | `python scripts/analyze_experiment.py --model otto` and `--model piecewise` | `results/{ot,pw}_summary.{json,md}`, figures 11-12, 15-16 | ~3 min |
| Theory checks | `python scripts/synthetic_fronts.py`, `python scripts/support_geometry.py`, `python scripts/linear_response.py`, `python scripts/gap_sensitivity.py` | `results/*.json`, figures 10, 14 | ~5 min |
| Physical cost of a linear figure of merit | `python scripts/physical_cost.py` | `results/physical_cost.{json,npz}`, figure 13 | ~3 min |
| Paper figures | `python paper/make_figures.py`, `python paper/make_saddle_figure.py` | `paper/figs/` | ~1 min |

The raw runs used in the paper are included in `results/`, so every analysis can be rerun without
repeating the optimization. `results/archive/preliminary_31k/` holds an earlier, shorter piecewise run
(10 seeds, 31,500 evaluations) kept for reference.

## Repository layout

```
hidden_front/      library (models, problems, MOEA/D, support test)
scripts/           experiments and analyses
tests/             unit tests (model vs. Monte Carlo and finite differences, MOEA/D, support test)
results/           raw runs and summaries
figures/           figures produced by the scripts
paper/             manuscript for an evolutionary-computation venue (IEEEtran) and its figures
paper/physics-draft/   earlier physics-oriented draft (REVTeX)
docs/proposition.md    technical note: statements, proofs sketch and evidence
docs/course-notes-es/  study guide for the thermodynamics course (Spanish)
talks/             course presentation and script (Spanish); earlier versions in talks/archive/
```

## Models and settings

- **Single-qubit Otto engine.** $T_h = 1$, $T_c = 0.5$ ($\eta_C = 0.5$), relaxation rate $\gamma = 1$,
  gap-change duration $0.05$. Decision variables: $\omega_h\in[0.1,10]$,
  $\omega_c/\omega_h\in[T_c/T_h,1]$, $\log_{10}\tau_{h,c}\in[-2,1]$. A realization with a flux-tunable
  superconducting qubit was proposed (as a refrigerator) by Karimi and Pekola (arXiv:1610.02776).
- **Piecewise quantum-dot engine.** Erdman et al., Phys. Rev. Research 5, L022017 (2023): $\beta_H=1$,
  $\beta_C=2$, $E_0=2.5$, $u\in[0.2,1.1]$, 8 control steps, 17 variables.
- **Algorithms.** NSGA-II, NSGA-III, SMS-EMOA and RVEA from pymoo 0.6.2; MOEA/D, MOEA/D-AWA and
  MOEA/D-WS from `hidden_front.moead`. Population 105 (Das-Dennis, 13 divisions), SBX and polynomial
  mutation with distribution index 20.
- **Metrics.** Hypervolume with reference point 1.1 and IGD+ in coordinates normalized by the reference
  front; two-sided Mann-Whitney tests against MOEA/D-AWA with Holm correction.

### MOEA/D-AWA implementation choices

Details that Qi et al. (2014) leave open are fixed explicitly in `hidden_front/moead.py`: sparsity is the
product of the distances to the $m$ nearest neighbours; 5% of the weights are adjusted every `wag`
generations after 80% of the run; the external population is capped at $1.5N$; $n_r = 2$,
$\delta = 0.9$ and a neighbourhood of $\lceil 0.1N\rceil$. The Tchebycheff function is **normalized** by
the ideal and nadir points of the population, which is necessary because the objectives differ by
orders of magnitude.

### Algorithm labels

| Label | Meaning |
|---|---|
| MOEA/D | normalized Tchebycheff, uniform weights |
| MOEA/D-WT | normalized Tchebycheff with WS-transformed weights (ablation, baseline runs only) |
| MOEA/D-AWA | MOEA/D-WT plus adaptive weight adjustment |
| MOEA/D-WS | linear weighted-sum scalarization (the baseline of the paper) |

## Citation

The manuscript is in preparation (`paper/main.tex`). Until it is published, please cite this repository.

## License

The code is released under the [MIT License](LICENSE).
