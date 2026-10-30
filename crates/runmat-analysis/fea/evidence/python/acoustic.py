"""Independent Python reproduction of the local plane-wave radiation fixture (unrotated)."""
import sys
import numpy as np
from j2front import mesh, shape, KEAST, EDGES

def mean_wave(xs, k):
    h = np.zeros(32, dtype=float); h[0] = 1.0
    for x in xs:
        for n in range(1, 32): h[n] += x * h[n - 1]
    coef = np.array([1.0, 0.0]); mean = np.zeros(2)
    for n in range(32):
        if n > 0: coef = np.array([coef[1] * k / n, -coef[0] * k / n])
        m = 6.0 * h[n] / ((n + 1) * (n + 2) * (n + 3)); mean += coef * m
    return mean[0] + 1j * mean[1]

# Dunavant degree 5 on triangle (barycentric, weight normalized to 1)
a1, b1 = 0.059715871789770, 0.470142064105115
a2, b2 = 0.797426985353087, 0.101286507323456
DUN = [([1/3, 1/3, 1/3], 0.225)]
DUN += [([a1, b1, b1], 0.132394152788506), ([b1, a1, b1], 0.132394152788506), ([b1, b1, a1], 0.132394152788506)]
DUN += [([a2, b2, b2], 0.125939180544827), ([b2, a2, b2], 0.125939180544827), ([b2, b2, a2], 0.125939180544827)]
TRI6_EDGES = [(0, 1), (1, 2), (2, 0)]

def tri_shape(L, quad):
    if not quad: return np.array(L)
    return np.array([L[i] * (2 * L[i] - 1) for i in range(3)] + [4 * L[a] * L[b] for a, b in TRI6_EDGES])

def solve(cells, quad, freq_base, amp, rho=3.0, c=2.0, kappa_override=None):
    f = freq_base * c / 2; w = 2 * np.pi * f; k = w / c
    X, tets = mesh(cells, quad)
    nn = len(X)
    A = np.zeros((nn, nn), dtype=complex)
    for t in tets:
        P = X[t[:4]]
        J = np.array([P[1] - P[0], P[2] - P[0], P[3] - P[0]]).T
        Ji = np.linalg.inv(J); det = np.linalg.det(J)
        for L, wq in KEAST:
            N, dNdL = shape(L, quad)
            G = (dNdL[:, 1:] - dNdL[:, [0]]) @ Ji
            A[np.ix_(t, t)] += (G @ G.T / rho - w * w / (rho * c * c) * np.outer(N, N)) * wq * det
    # radiation faces at x=1
    for t in tets:
        c4 = t[:4]
        on = [i for i in range(4) if np.isclose(X[c4[i], 0], 1.0)]
        if len(on) != 3: continue
        nodes = [c4[i] for i in on]
        if quad:
            for a, b in TRI6_EDGES:
                ia, ib = on[a], on[b]
                k_ = [kk for kk, e in enumerate(EDGES) if set(e) == {ia, ib}][0]
                nodes.append(t[4 + k_])
        P = X[[c4[i] for i in on]]
        area = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
        for L, wq in DUN:
            N = tri_shape(L, quad)
            A[np.ix_(nodes, nodes)] += 1j * w / (rho * c) * np.outer(N, N) * wq * area
    fixed = np.isclose(X[:, 0], 0.0); free = ~fixed
    p = np.zeros(nn, dtype=complex); p[fixed] = amp
    rhs = -A[np.ix_(free, fixed)] @ p[fixed]
    p[free] = np.linalg.solve(A[np.ix_(free, free)], rhs)
    Z = rho * c
    perr = np.sqrt(np.mean(np.abs(p - amp * np.exp(-1j * k * X[:, 0])) ** 2))
    verr = 0.0; gal = 0.0; interp = 0.0
    pex = amp * np.exp(-1j * k * X[:, 0])
    for t in tets:
        P = X[t[:4]]
        J = np.array([P[1] - P[0], P[2] - P[0], P[3] - P[0]]).T
        Ji = np.linalg.inv(J)
        # mean gradient: average over keast points of grad
        g = np.zeros(3, dtype=complex); gi = np.zeros(3, dtype=complex)
        for L, wq in KEAST:
            N, dNdL = shape(L, quad)
            G = (dNdL[:, 1:] - dNdL[:, [0]]) @ Ji
            g += G.T @ p[t] * wq * 6; gi += G.T @ pex[t] * wq * 6
        v = -g / (1j * w * rho); vi = -gi / (1j * w * rho)
        vex = np.array([amp * mean_wave(P[:, 0], k) / Z, 0, 0])
        verr += np.sum(np.abs(v - vex) ** 2); gal += np.sum(np.abs(v - vi) ** 2); interp += np.sum(np.abs(vi - vex) ** 2)
    n = len(tets)
    return perr, np.sqrt(verr / n), np.sqrt(gal / n), np.sqrt(interp / n)

if __name__ == "__main__":
    quad = sys.argv[1] == "tet10"; fb = float(sys.argv[2])
    cells = [int(c) for c in sys.argv[3].split(",")]
    amp = 1.0 + 0.4j
    prev = None
    for cc in cells:
        r = solve(cc, quad, fb, amp)
        print(cc, "pressure=%.6e velocity=%.6e galerkin=%.6e interp=%.6e" % r, "" if prev is None else "ratios p=%.3f v=%.3f" % (prev[0] / r[0], prev[1] / r[1]))
        prev = r
