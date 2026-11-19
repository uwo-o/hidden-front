# Technical note: geometry of fronts with a ratio objective

This note summarizes the theory behind the paper and the evidence for it. The formal statements and
proofs are in `paper/main.tex` (Section III).

## General statement

Let $a>0$ be an extensive output to maximize (e.g. power $P$), $r$ a ratio to maximize (e.g.
efficiency $\eta$) and $c$ a cost to minimize (e.g. power fluctuations $\Delta P$). A front is
**product-structured** if it is the set

$$\mathcal{F}_r=\{(-a,\,-r,\,a\,\varphi(r)) : (a,r)\in D\},$$

with $D$ two-dimensional and $\varphi>0$, $\varphi'>0$.

1. **Ratio formulation (Theorem 1).** No interior point of $\mathcal{F}_r$ is supported: no weighted
   sum $w\cdot f$ with $w\ge 0$ is minimized there. The Hessian of $g(a,r)=a\varphi(r)$ is
   $$\nabla^2 g=\begin{pmatrix}0 & \varphi'\\ \varphi' & a\varphi''\end{pmatrix},\qquad
   \det\nabla^2 g=-\varphi'^2<0,$$
   so the surface is a saddle at every interior point, whatever $\varphi''$ is.

2. **Extensive formulation (Theorem 2).** Let $b=a\,s(r)$ be an extensive cost with $s>0$ strictly
   decreasing (waste per unit output, entropy produced per unit work). Then
   $c=a\,\chi(b/a)$ with $\chi=\varphi\circ s^{-1}$, the perspective of $\chi$. The two formulations have
   the same Pareto set. If $\chi$ is convex (and the image of $D$ is convex), every point is
   supported with nonnegative weights; if $\chi''<0$ on an interval, the corresponding interior points
   are not.

3. **Degeneracy (Corollary 1).** In the convex case $c$ is positively homogeneous of degree one, hence
   linear along rays $b/a=\text{const}$. Each supporting weighted sum is minimized along a whole
   segment, so weighted-sum decomposition crowds its solutions.

4. **Free scale (Theorem 3).** If the decision vector is (lambda, z) with a scale lambda that multiplies
   every extensive objective and leaves intensive ones unchanged, a supported point with interior scale
   requires w_E . e(z*) = 0 and is tied along the whole scale segment, and z* must be critical for the
   weighted intensive objectives. With one ratio whose gradient never vanishes, a weighted sum that
   rewards the ratio returns only designs at the scale bounds.

5. **Dimension.** Theorem 1 needs a two-dimensional $D$. A one-dimensional front lying on the saddle
   can be fully supported.

## The physical case: TUR saturation

For a heat engine between baths at $T_h>T_c$ ($\eta_C=1-T_c/T_h$), saturation of the thermodynamic
uncertainty relation $\Delta P\,\Sigma\ge 2P^2$ gives a product structure:

$$\Delta P=P\,\varphi(\eta),\qquad \varphi(\eta)=\frac{2T_c\,\eta}{\eta_C-\eta}.$$

The entropy production rate $\Sigma=P(\eta_C-\eta)/(\eta T_c)$ is the extensive reformulation, with
$\chi(u)=2/u$ convex, so $\Delta P=2P^2/\Sigma$:

- in $(-P,-\eta,\Delta P)$ the front is a saddle and no interior point is supported;
- in $(-P,\Delta P,\Sigma)$ it is a convex trough, fully supported but degenerate along rays.

The symbolic Hessians are checked in `scripts/support_geometry.py`.

## Evidence

| Experiment | Script | Result |
|---|---|---|
| Synthetic product-structured fronts, four profiles $\varphi$ | `scripts/synthetic_fronts.py` | 86-93% unsupported with the ratio objective; 0% with the extensive one when $\chi$ is convex; 78% when $\chi$ is concave |
| Synthetic TUR-saturating surface | `scripts/support_geometry.py` | 85-86% unsupported in $(P,\eta,\Delta P)$; 0% in $(P,\Delta P,\Sigma)$ |
| Single-qubit Otto engine (saturates the TUR, two-dimensional front) | `scripts/analyze_experiment.py --model otto` | 92% unsupported with $\eta$; 2% with $\Sigma$. The weighted-sum HV ceiling is 0.9946 vs. 1.0805 for the reference front with $\eta$, and 1.2115 vs. 1.2115 with $\Sigma$ |
| Physical cost for a weighted-sum designer | `scripts/physical_cost.py` | With $\eta$: 11% of the front reachable, median loss 14%. With $\Sigma$: median loss 0% |
| Linear response (one-dimensional front) | `scripts/linear_response.py` | The front is a curve on the TUR surface and is 100% supported |
| Reverse-osmosis plant sizing (free scale, 4 objectives) | `scripts/analyze_extended.py --model ro` | 43% unsupported with recovery, 24% with brine; ceiling gap 2.6% vs 0.2%; every MOEA/D-WS solution at n = 1 or 100 in both formulations |
| Tolerance and resolution | `scripts/structure_robustness.py` | Interior points: 95% unsupported at 1e-3, 100% at 1e-5, independent of resolution |
| Breaking homogeneity, c = a^(1+kappa) phi(r) | `scripts/structure_robustness.py` | Ratio formulation 81-86% unsupported for kappa in [-0.75, 1]; extensive formulation nonconvex for kappa < 0 |
| Piecewise control (violates the TUR, $\xi$ up to 4) | `scripts/analyze_experiment.py --model piecewise` | Nearly convex in both formulations: the weighted-sum ceiling is within 0.1-0.2% of the reference HV |

## Decomposition of the weighted-sum loss

The HV of the supported part of a reference front is an upper bound (ceiling) on what any weighted-sum
method can achieve. It splits the HV loss of MOEA/D-WS into

- **reachability loss** = HV(reference) - ceiling, which no choice of weights can recover;
- **distribution loss** = ceiling - HV(MOEA/D-WS), due to how the weights spread the solutions.

On the Otto engine, the loss is mostly reachability with $\eta$ (0.086 + 0.028) and entirely
distribution with $\Sigma$ (0.0001 + 0.089), as Theorems 1-2 and Corollary 1 predict.

## Limitations

- Real fronts are only approximately product-structured. On the Otto engine $\xi$ ranges between about
  0.95 and 1.005; cyclic engines can slightly exceed the steady-state bound.
- When the structure is broken strongly (piecewise control or kappa >= 2), the effect vanishes; economies
  of scale (kappa < 0) also make the extensive formulation nonconvex.
- The support test and the ceiling use a discrete reference front, so they carry an error of the order
  of the tolerance ($10^{-3}$ in normalized units).
