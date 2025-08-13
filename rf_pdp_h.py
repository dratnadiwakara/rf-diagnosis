# Minimal RF + PDP + Friedman H interaction index
# ------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

rng = check_random_state(42)

# 1) Synthetic data with a real interaction between x1 and x2
n = 1500
x1 = rng.uniform(-2, 2, size=n)
x2 = rng.uniform(-2, 2, size=n)
x3 = rng.normal(size=n)                   # nuisance feature (no interaction)
y  = 1.0 + 1.5*np.sin(x1) + 0.8*x2 + 2.0*(x1*x2) + rng.normal(0, 0.5, size=n)
X  = np.column_stack([x1, x2, x3])

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

rf = RandomForestRegressor(n_estimators=180, min_samples_leaf=2,
                           random_state=42, n_jobs=1).fit(Xtr, ytr)

# Small evaluation subset for fast PDPs
idx   = rng.choice(Xte.shape[0], size=min(200, Xte.shape[0]), replace=False)
Xeval = Xte[idx]

def pdp_1d(model, X_eval, j, grid):
    out = np.zeros(len(grid))
    for a, v in enumerate(grid):
        Xmod = X_eval.copy()
        Xmod[:, j] = v
        out[a] = model.predict(Xmod).mean()
    return out

def pdp_2d(model, X_eval, j, k, grid_j, grid_k):
    P = np.zeros((len(grid_j), len(grid_k)))
    for a, vj in enumerate(grid_j):
        for b, vk in enumerate(grid_k):
            Xmod = X_eval.copy()
            Xmod[:, j] = vj
            Xmod[:, k] = vk
            P[a, b] = model.predict(Xmod).mean()
    return P

# Grids within interior percentiles
q_low, q_high, gsize = 0.02, 0.98, 24
g1 = np.quantile(Xeval[:, 0], np.linspace(q_low, q_high, gsize))
g2 = np.quantile(Xeval[:, 1], np.linspace(q_low, q_high, gsize))

# 2-D PDP and 1-D PDPs
P   = pdp_2d(rf, Xeval, j=0, k=1, grid_j=g1, grid_k=g2)
PD1 = pdp_1d(rf, Xeval, j=0, grid=g1)
PD2 = pdp_1d(rf, Xeval, j=1, grid=g2)

# Friedman H index for (x1, x2)
A   = P - PD1[:, None] - PD2[None, :]
H   = np.sqrt(np.var(A) / np.var(P))

print("H(x1,x2) =", float(H))

# Sanity check with an ANOVA-style double-centering (equivalent here)
row_m, col_m, grand_m = P.mean(axis=1, keepdims=True), P.mean(axis=0, keepdims=True), P.mean()
H_dc = np.sqrt(np.var(P - row_m - col_m + grand_m) / np.var(P))
print("H (double-centered) =", float(H_dc))

# Contrast: non-interacting pair (x1, x3) → H near zero
g3   = np.quantile(Xeval[:, 2], np.linspace(q_low, q_high, gsize))
P13  = pdp_2d(rf, Xeval, j=0, k=2, grid_j=g1, grid_k=g3)
rm13, cm13, gm13 = P13.mean(1, keepdims=True), P13.mean(0, keepdims=True), P13.mean()
H_13 = np.sqrt(np.var(P13 - rm13 - cm13 + gm13) / np.var(P13))
print("H(x1,x3) ≈", float(H_13))

# 2-D PDP surface (one chart per figure; default colormap)
plt.figure(figsize=(6,5))
cp = plt.contourf(g2, g1, P, levels=16)
plt.xlabel("x2")
plt.ylabel("x1")
plt.title("2D Partial Dependence: (x1, x2)")
plt.colorbar(cp)
plt.tight_layout()
plt.show()
