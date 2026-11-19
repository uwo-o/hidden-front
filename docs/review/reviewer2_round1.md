# Reviewer 2 — Round 1 (self-review before submission)

**Manuscript:** "Ratio Objectives Hide the Pareto Front from Weighted-Sum Decomposition: Theory,
Diagnostics, and a Quantum Heat Engine Case Study"

**Recommendation:** Major revision (leaning reject in current form)

## Summary

The paper argues that fronts of the form (−a, −r, a·φ(r)) are saddles with no supported interior
point, that replacing the ratio r with an extensive cost turns them into a perspective function, and
illustrates this with a quantum Otto engine whose front saturates a thermodynamic uncertainty relation.

## Major concerns

**M1. The theory is elementary and its scope is unclear.** Theorem 1 is a two-line Hessian computation,
and Theorem 2 restates the convexity of perspective functions (Boyd & Vandenberghe). The paper must show
why this structure matters *beyond one physics model*. Where else does c = a·φ(r) arise? The authors
say "common in engineering" without a single engineering example.

**M2. A single real-world instance, from a niche field.** The quantum Otto engine is exotic for an EC
audience, and its product structure relies on TUR saturation, which is approximate (ξ ∈ [0.95, 1.005]).
I need at least one more real-world problem, from a mainstream engineering domain, where the
structure arises for a different reason.

**M3. MOEA/D-WS is a straw man.** Weighted sums are known to fail on nonconvex fronts (Das & Dennis
1997). The EC community has many scalarizations designed for nonconvex fronts: PBI, L_p
scalarizations, and the localized weighted sum (LWS, Wang et al. 2018). Does the saddle defeat them?
Without these baselines the empirical part shows only what is already known.

**M4. Is the loss an artefact of budget or population size?** The reachability loss could shrink
with more generations or a larger N. Show that it does not, and that the distribution loss in the
extensive formulation does or does not.

**M5. The diagnostics depend on discretization.** Theorem 1 says *no* interior point is supported,
yet the synthetic fronts report 86-93% and the Otto front 91%. Explain the gap. Show how the
unsupported fraction depends on the tolerance (10⁻³ is arbitrary) and on the resolution of the
reference front.

**M6. The HV "ceiling" claim is imprecise.** A finite-budget weighted-sum run need not return
weighted-sum minimizers; its population can contain points in the unsupported region (Table II
reports 2%). State exactly what the ceiling bounds.

**M7. The open question in "Limitations" is the interesting one.** How does the reachability loss
behave when the structure holds only approximately? A controlled experiment that interpolates between
exact product structure and no structure is needed.

**M8. Statistics.** Report effect sizes (e.g., Vargha–Delaney Â₁₂), not only p-values that all saturate
at the minimum for n = 30.

## Minor concerns

- m1. Theorem 1 uses only second-order *local* conditions; state that it excludes local support, and
  discuss what this implies for localized methods.
- m2. Related work omits adaptive weighted sums (Kim & de Weck), L_p scalarizations, LWS, and
  front-shape-aware MOEA/D variants.
- m3. Verify bibliographic entries (e.g., Li 1996).
- m4. Generalize beyond three objectives or state the restriction.
