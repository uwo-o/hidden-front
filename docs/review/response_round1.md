# Response to Reviewer 2 (round 1) and Reviewer 2's second-round assessment

## Point-by-point response

**M1 — Elementary theory, unclear scope.**
We added Theorem 3 (free scale, Sec. III-C). For any number of objectives, a design with a free size
variable and an intensive objective yields the same obstruction: a weighted sum with positive weight on
the ratio is minimized only at the bounds of the scale (Corollary 2). In an all-extensive formulation,
every supported point with interior scale lies on a whole segment of ties. This links the structure to a
mechanism that is ubiquitous in engineering design (number of parallel units, area, capacity) and
explains why product structure is common.

**M2 — A single, niche real-world instance.**
We added a second case study from a mainstream domain: the sizing of a seawater reverse-osmosis plant
with 4 objectives (production, recovery, pump power, number of vessels) and an extensive reformulation
(brine discharge instead of recovery). The model has a closed-form solution, validated against ODE
integration to 1e-11, and 9 unit tests. The structure comes from the free scale, not from thermodynamic
uncertainty relations. Results: recovery hides 44% of the front, but the reachability loss is modest
(2.6% of HV). The dominant effect, predicted by Theorem 3(i), is that every MOEA/D-WS solution lies at
n = 1 or n = 100 in both formulations, which costs two thirds of the HV.

**M3 — MOEA/D-WS is a straw man.**
We added PBI (θ = 5), L2, L4 and the localized weighted sum (LWS), all in the same code base, with 30
runs each on both models and both formulations.
- PBI and Lp escape the saddle, and Lp improves monotonically with p toward Tchebycheff.
- LWS depends on the width of its cones. With the default half-angle, its median HV is below the
  weighted-sum ceiling and only 27% of its runs exceed it. Wider cones (2x, 4x) make it worse, and no run
  exceeds the ceiling. Narrower cones (0.5x, 0.25x) let it escape: every run exceeds the ceiling, and the
  median HV reaches 1.0342. We explain why: on a saddle, the optimum inside each cone lies on the cone
  boundary, so narrow cones act as a direction constraint.
- On RO, PBI is the best method in both formulations.

**M4 — Budget and population.**
Five times the budget leaves MOEA/D-WS unchanged (0.9671 vs 0.9670 in the ratio formulation; 1.1222 vs
1.1220 in the extensive one). With N = 231, MOEA/D-WS reaches 0.9791 in the ratio formulation, still
below the ceiling (0.9946) in every run, and stays at 1.1182 in the extensive one, with 128 and 146
distinct solutions out of 231.

**M5 — Discretization of the diagnostics.**
New Fig. 6a: for interior test points, the unsupported fraction is 66%, 95%, 99.7% and 100% at
tolerances 1e-2, 1e-3, 1e-4 and 1e-5. It is independent of the resolution of the comparison front
(400 to 6400 points). The 86–93% of Table I comes from points near the boundary of D. Tolerance 1e-3 is
conservative. The extended tables also report unsupported fractions at four tolerances.

**M6 — Imprecise ceiling.**
Sec. IV-B now states that the ceiling bounds the HV of any set of weighted-sum minimizers. A
finite-budget population may hold non-optimal points in the unsupported region, so we report that
fraction (0–4% for MOEA/D-WS).

**M7 — Approximate structure.**
New κ family, c = a^(1+κ) φ(r) (Fig. 6b,c):
- The ratio formulation stays 81–86% unsupported for κ ∈ [−0.75, 1].
- Economies of scale (κ < 0) make the extensive formulation nonconvex (46–78% unsupported, small
  gaps).
- Strong diseconomies (κ ≥ 2) convexify the front.

**M8 — Effect sizes.**
Vargha–Delaney Â12 is reported for every comparison. Almost all comparisons are completely separated
(Â12 ∈ {0, 1}).

**Minor points.**
- m1: Theorem 1 is stated locally, and its consequences for localized methods are discussed.
- m2: The related work now covers adaptive weighted sums, Lp/PaS, LWS, PBI, front-shape studies and
  physics work on front convexity.
- m3: References were verified: Li 1996 (JOTA 88(1)); Qi 2014 (Evol. Comput. 22(2):231–264); Cheng 2016
  (TEVC 20(5):773–791); Zhu et al. is 2009, not 2008. Uncited entries were removed.
- m4: Theorem 3 holds for any m. The RO study has m = 4.

**Corrections made during the revision.**
- The L_p scalarization was first implemented as (Σ λ_i f_i^p)^(1/p), which does not tend to
  Tchebycheff. It is now (Σ (λ_i f_i)^p)^(1/p), with a unit test, and those runs were repeated.
- SMS-EMOA is omitted on the four-objective RO problem because exact hypervolume-contribution selection
  is too expensive there. This is stated in the paper.

---

## Reviewer 2 — Round 2

**Recommendation:** Accept after minor revision (all items below are addressed in the text or justified).

The revision addresses all major concerns with new theory, a second real-world problem, the relevant
decomposition baselines and robustness studies. The central claims are now supported by evidence that
goes beyond one physics model, and the paper is honest about magnitudes: the reachability loss is large
on the Otto engine and small on the RO plant, while the free-scale degeneracy is large on the latter.

Remaining minor issues, which the authors may address at the camera-ready stage or in the response:

1. **LWS cone angle.** *Addressed in this round.* The half-angle is now stated as an implementation
   choice, and a sensitivity study (×0.25 to ×4, 10 runs each) shows that LWS escapes the saddle only
   with narrow cones. The paper's guidelines were corrected accordingly.
2. **RO realism.** *Addressed.* The paper now states that concentration polarization, pressure drop
   and flux limits change the numbers but not the free-scale structure, so Theorem 3 still applies.
3. **Piecewise model.** *Addressed.* The paper states that the extra baselines were not run there
   because neither theorem applies.
4. **Venue.** The contribution fits IEEE TEVC, Evolutionary Computation (MIT Press) or Swarm and
   Evolutionary Computation. The 10-page length suits a regular paper.

None of these affects the conclusions.
