"""MOEA/D, MOEA/D with WS-transformed weights, and MOEA/D-AWA (Qi et al., Evol. Comput. 22(2), 2014).

A single code base with four modes:
    MOEAD(problem)                        -> MOEA/D (Zhang & Li 2007), normalized Tchebycheff
    MOEAD(problem, ws=True)               -> ablation: WS-transformed initial weights only
    MOEAD(problem, ws=True, awa=True)     -> MOEA/D-AWA
    MOEAD(problem, scalarization="ws")    -> MOEA/D with a linear weighted sum (MOEA/D-WS)

Further scalarizations (all on normalized objectives f_hat = (f - z*) / (z_nad - z*)):
    scalarization="lp", p=...  -> L_p scalarization (sum_i (lam_i f_hat_i)^p)^(1/p) (Giagkiozis & Fleming 2015);
                                  p = 1 is the weighted sum, p -> inf approaches Tchebycheff
    scalarization="pbi", theta=5 -> penalty-based boundary intersection (Zhang & Li 2007)
    scalarization="lws"        -> localized weighted sum (Wang et al., IEEE TEVC 22(1), 2018): the weighted
                                  sum is used only inside a hypercone around lam_i whose half-angle is the
                                  mean angle between lam_i and its m nearest weight vectors; outside the cone
                                  the value is a large constant plus the angle, so that solutions closer to
                                  the cone are preferred

Details of MOEA/D-AWA that the original paper leaves open, fixed here explicitly:
  * Sparsity: SL(ind, pop) = prod_{i=1}^{m} d_i, with d_i the Euclidean distance to
    the i-th nearest neighbour (k = m), as in reference implementations.
  * wag (adjustment period) in generations; default 20.
  * nr = 2 (maximum number of replacements per offspring).
  * External population (EP): nondominated archive of everything evaluated, capped
    at 1.5 N; when exceeded, the member with the lowest sparsity is removed iteratively.

Deliberate modification with respect to the original paper:
  * NORMALIZED Tchebycheff: f_hat = (f - z*) / (z_nad - z*), with z* the observed
    ideal point and z_nad the maximum of the current population. Sparsity and new
    weights are also computed in the normalized space. Without this, objectives on
    very different scales (eta ~ 0.3 vs dP ~ 1e-2) bias both the decomposition and
    the density measure.
"""
from __future__ import annotations

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

EPS = 1e-6
_NDS = NonDominatedSorting()


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def ws_transform(lam, eps=EPS):
    """WS-transformation: lam'_i = (1/(lam_i+eps)) / sum_k 1/(lam_k+eps). Self-inverse as eps -> 0."""
    inv = 1.0 / (np.asarray(lam, dtype=float) + eps)
    return inv / inv.sum(axis=-1, keepdims=True)


def sparsity(A, B=None, k=None):
    """Sparsity level of each row of A with respect to the set B (product of the
    distances to the k nearest neighbours). If B is None, A is used, excluding
    each point from its own neighbourhood."""
    self_ref = B is None
    B = A if self_ref else B
    k = A.shape[1] if k is None else k
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    if self_ref:
        np.fill_diagonal(D, np.inf)
    k = min(k, D.shape[1] - (1 if self_ref else 0))
    if k <= 0:
        return np.full(len(A), np.inf)
    nn = np.partition(D, k - 1, axis=1)[:, :k]
    return np.prod(nn, axis=1)


def nondominated_mask(F):
    idx = _NDS.do(F, only_non_dominated_front=True)
    mask = np.zeros(len(F), dtype=bool)
    mask[idx] = True
    return mask


def dominates_any(F_a, F_b):
    """Mask over F_b: True where some point of F_a dominates the point of F_b."""
    le = (F_a[:, None, :] <= F_b[None, :, :]).all(-1)
    lt = (F_a[:, None, :] < F_b[None, :, :]).any(-1)
    return (le & lt).any(0)


# --------------------------------------------------------------------------- #
# Variation operators (SBX + polynomial mutation, Deb)
# --------------------------------------------------------------------------- #
def sbx(p1, p2, xl, xu, eta, rng, prob_var=0.5):
    c1, c2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        if rng.random() > prob_var or abs(p1[i] - p2[i]) < 1e-14:
            continue
        y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
        lo, hi = xl[i], xu[i]
        u = rng.random()

        def child(beta_bound):
            alpha = 2.0 - beta_bound ** -(eta + 1.0)
            if u <= 1.0 / alpha:
                betaq = (u * alpha) ** (1.0 / (eta + 1.0))
            else:
                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta + 1.0))
            return betaq

        bq1 = child(1.0 + 2.0 * (y1 - lo) / (y2 - y1))
        bq2 = child(1.0 + 2.0 * (hi - y2) / (y2 - y1))
        v1 = 0.5 * ((y1 + y2) - bq1 * (y2 - y1))
        v2 = 0.5 * ((y1 + y2) + bq2 * (y2 - y1))
        v1, v2 = np.clip(v1, lo, hi), np.clip(v2, lo, hi)
        if rng.random() < 0.5:
            v1, v2 = v2, v1
        c1[i], c2[i] = v1, v2
    return c1, c2


def poly_mutation(x, xl, xu, eta, rng, pm=None):
    y = x.copy()
    n = len(x)
    pm = 1.0 / n if pm is None else pm
    for i in range(n):
        if rng.random() >= pm:
            continue
        lo, hi = xl[i], xu[i]
        d1, d2 = (y[i] - lo) / (hi - lo), (hi - y[i]) / (hi - lo)
        u = rng.random()
        mp = 1.0 / (eta + 1.0)
        if u < 0.5:
            xy = 1.0 - d1
            val = 2.0 * u + (1.0 - 2.0 * u) * xy ** (eta + 1.0)
            dq = val ** mp - 1.0
        else:
            xy = 1.0 - d2
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta + 1.0)
            dq = 1.0 - val ** mp
        y[i] = np.clip(y[i] + dq * (hi - lo), lo, hi)
    return y


# --------------------------------------------------------------------------- #
# MOEA/D with optional AWA
# --------------------------------------------------------------------------- #
class MOEAD:
    def __init__(self, problem, n_partitions=13, max_gen=500, ws=False, awa=False,
                 T_frac=0.1, delta=0.9, nr=2, eta_c=20.0, eta_m=20.0,
                 rate_evol=0.8, wag=20, rate_update_weight=0.05, ep_factor=1.5,
                 normalize=True, hist_every=10, seed=0, scalarization="tch", p=2.0, theta=5.0, cone_factor=1.0):
        if scalarization not in ("tch", "ws", "lp", "pbi", "lws"):
            raise ValueError("scalarization must be one of 'tch', 'ws', 'lp', 'pbi', 'lws'")
        if scalarization != "tch" and (ws or awa):
            raise ValueError("WS transformation and AWA are only used with the Tchebycheff scalarization")
        self.scalarization = scalarization
        self.p, self.theta = float(p), float(theta)
        self.cone_factor = float(cone_factor)
        if awa and not ws:
            raise ValueError("AWA requires WS-transformed weights (ws=True).")
        self.problem = problem
        self.m = problem.n_obj
        self.W0 = get_reference_directions("das-dennis", self.m, n_partitions=n_partitions)
        self.N = len(self.W0)
        self.max_gen = max_gen
        self.ws, self.awa = ws, awa
        self.T = max(2, int(np.ceil(T_frac * self.N)))
        self.delta, self.nr = delta, nr
        self.eta_c, self.eta_m = eta_c, eta_m
        self.rate_evol, self.wag = rate_evol, wag
        self.nus = max(1, int(round(rate_update_weight * self.N)))
        self.ep_cap = int(np.ceil(ep_factor * self.N))
        self.normalize = normalize
        self.hist_every = hist_every
        self.rng = np.random.default_rng(seed)
        self.xl = np.asarray(problem.xl, dtype=float)
        self.xu = np.asarray(problem.xu, dtype=float)
        self.n_eval = 0
        self.n_adjust = 0

    # ---------------- evaluation and scalarization ---------------- #
    def _eval(self, X):
        F = self.problem.evaluate(np.atleast_2d(X), return_values_of=["F"])
        self.n_eval += len(np.atleast_2d(X))
        return F

    def _scale(self):
        if not self.normalize:
            return np.ones(self.m)
        s = self.nadir - self.z
        return np.where(s > 1e-12, s, 1e-12)

    def _norm(self, F):
        return (F - self.z) / self._scale()

    def _g(self, F, lam, cone=None):
        """Scalarizing function on normalized objectives. F (..., m), lam (..., m).
        cone: half-angles of the subproblems (only for "lws"), broadcast like lam[..., 0]."""
        fh = self._norm(F)
        sc = self.scalarization
        if sc == "ws":
            return np.sum(lam * fh, axis=-1)
        if sc == "lp":
            return np.sum((lam * np.maximum(fh, 0.0)) ** self.p, axis=-1) ** (1.0 / self.p)
        if sc == "pbi":
            u = lam / np.linalg.norm(lam, axis=-1, keepdims=True)
            d1 = np.sum(fh * u, axis=-1)
            d2 = np.linalg.norm(fh - d1[..., None] * u, axis=-1)
            return d1 + self.theta * d2
        if sc == "lws":
            u = lam / np.linalg.norm(lam, axis=-1, keepdims=True)
            fp = np.maximum(fh, 0.0)
            nf = np.linalg.norm(fp, axis=-1)
            cosang = np.sum(fp * u, axis=-1) / np.where(nf > 1e-12, nf, 1e-12)
            ang = np.arccos(np.clip(cosang, -1.0, 1.0))
            inside = (ang <= cone) | (nf <= 1e-12)
            return np.where(inside, np.sum(lam * fh, axis=-1), 1e6 + ang)
        return np.max(lam * fh, axis=-1)

    def _cones(self):
        """LWS half-angles: mean angle between each weight vector and its m nearest neighbours."""
        u = self.lam / np.linalg.norm(self.lam, axis=1, keepdims=True)
        A = np.arccos(np.clip(u @ u.T, -1.0, 1.0))
        np.fill_diagonal(A, np.inf)
        k = min(self.m, len(u) - 1)
        self.cone = self.cone_factor * np.mean(np.sort(A, axis=1)[:, :k], axis=1)

    def _directions(self):
        """Solution-mapping vectors (used to build neighbourhoods)."""
        return ws_transform(self.lam) if self.ws else self.lam

    def _neighborhoods(self):
        d = self._directions()
        D = ((d[:, None, :] - d[None, :, :]) ** 2).sum(-1)
        T = min(self.T, len(d))
        self.B = np.argsort(D, axis=1)[:, :T]

    # ---------------- external population ---------------- #
    def _update_ep(self, X_new, F_new):
        X = np.vstack([self.EP_X, X_new]) if len(self.EP_X) else X_new
        F = np.vstack([self.EP_F, F_new]) if len(self.EP_F) else F_new
        _, uniq = np.unique(np.round(F, 12), axis=0, return_index=True)
        X, F = X[uniq], F[uniq]
        nd = nondominated_mask(F)
        X, F = X[nd], F[nd]
        if len(F) > self.ep_cap:
            Fn = self._norm(F)
            D = np.sqrt(((Fn[:, None] - Fn[None]) ** 2).sum(-1))
            np.fill_diagonal(D, np.inf)
            alive = np.ones(len(F), dtype=bool)
            k = self.m
            while alive.sum() > self.ep_cap:
                ids = np.flatnonzero(alive)
                sub = D[np.ix_(ids, ids)]
                sl = np.prod(np.partition(sub, k - 1, axis=1)[:, :k], axis=1)
                alive[ids[np.argmin(sl)]] = False
            X, F = X[alive], F[alive]
        self.EP_X, self.EP_F = X, F

    # ---------------- adaptive weight adjustment ---------------- #
    def _adjust_weights(self):
        # Algorithm 2, step 1: every subproblem takes the best individual of the population
        G = self._g(self.F[None, :, :], self.lam[:, None, :])        # G[j, i]
        best = np.argmin(G, axis=1)
        improve = G[np.arange(len(G)), best] < G[np.arange(len(G)), np.arange(len(G))]
        X_old, F_old = self.X.copy(), self.F.copy()
        self.X[improve] = X_old[best[improve]]
        self.F[improve] = F_old[best[improve]]

        # Algorithm 2, steps 2-3: delete nus overcrowded subproblems
        for _ in range(self.nus):
            sl = sparsity(self._norm(self.F))
            k = np.argmin(sl)
            self.X = np.delete(self.X, k, 0)
            self.F = np.delete(self.F, k, 0)
            self.lam = np.delete(self.lam, k, 0)

        # Algorithm 3: add nus subproblems in sparse regions using the EP
        keep = ~dominates_any(self.F, self.EP_F)
        EX, EF = self.EP_X[keep], self.EP_F[keep]
        for _ in range(self.nus):
            if len(EF) == 0:
                break
            sl = sparsity(self._norm(EF), self._norm(self.F))
            k = np.argmax(sl)
            fhat = np.maximum(self._norm(EF[k]), 0.0)
            lam_new = ws_transform(fhat)       # lam ~ 1/(f_hat - z_hat + eps)  (Theorem 2 of Qi et al.)
            self.X = np.vstack([self.X, EX[k]])
            self.F = np.vstack([self.F, EF[k]])
            self.lam = np.vstack([self.lam, lam_new])
            EX, EF = np.delete(EX, k, 0), np.delete(EF, k, 0)
        self._neighborhoods()
        self.n_adjust += 1

    # ---------------- main loop ---------------- #
    def run(self, verbose=False):
        rng = self.rng
        n_var = len(self.xl)
        self.lam = ws_transform(self.W0) if self.ws else np.maximum(self.W0, EPS)
        self._neighborhoods()
        self.cone = None
        if self.scalarization == "lws":
            self._cones()
        self.X = self.xl + rng.random((self.N, n_var)) * (self.xu - self.xl)
        self.F = self._eval(self.X)
        self.z = self.F.min(0)
        self.nadir = self.F.max(0)
        self.EP_X = np.empty((0, n_var))
        self.EP_F = np.empty((0, self.m))
        self._update_ep(self.X, self.F)
        self.history = []

        awa_start = int(self.rate_evol * self.max_gen)
        for gen in range(self.max_gen):
            children_X, children_F = [], []
            for i in rng.permutation(len(self.X)):
                if rng.random() < self.delta:
                    pool = self.B[i]
                else:
                    pool = np.arange(len(self.X))
                a, b = rng.choice(pool, 2, replace=False)
                c1, c2 = sbx(self.X[a], self.X[b], self.xl, self.xu, self.eta_c, rng)
                child = c1 if rng.random() < 0.5 else c2
                child = poly_mutation(child, self.xl, self.xu, self.eta_m, rng)
                fc = self._eval(child)[0]
                self.z = np.minimum(self.z, fc)
                children_X.append(child)
                children_F.append(fc)
                # neighbour update (at most nr replacements)
                cand = rng.permutation(pool)
                cone = None if self.cone is None else self.cone[cand]
                g_child = self._g(fc[None, :], self.lam[cand], cone)
                g_cur = self._g(self.F[cand], self.lam[cand], cone)
                replaced = 0
                for j, gc, gj in zip(cand, g_child, g_cur):
                    if gc <= gj:
                        self.X[j] = child
                        self.F[j] = fc
                        replaced += 1
                        if replaced >= self.nr:
                            break
            self.nadir = self.F.max(0)
            self._update_ep(np.array(children_X), np.array(children_F))

            if self.awa and gen >= awa_start and (gen - awa_start) % self.wag == 0:
                self._adjust_weights()

            if gen % self.hist_every == 0 or gen == self.max_gen - 1:
                self.history.append((gen, self.n_eval, self.F.copy()))
            if verbose and gen % 50 == 0:
                print(f"gen {gen:4d}  evals {self.n_eval}  |EP|={len(self.EP_F)}")
        return self

    @property
    def result_F(self):
        """Nondominated front of the final population (duplicates removed)."""
        F = np.unique(np.round(self.F, 14), axis=0)
        return F[nondominated_mask(F)]

    @property
    def result_X(self):
        mask = nondominated_mask(self.F)
        return self.X[mask]
