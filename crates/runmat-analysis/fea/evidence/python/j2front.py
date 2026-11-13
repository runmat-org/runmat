"""Independent Python reproduction of the moving-front J2 uniaxial-strain fixture.
Freudenthal Tet4/Tet10 slab, Ux-only (Uy=Uz=0 everywhere), Ux=0 at x=0, Keast15 points."""
import sys
import numpy as np

E, NU, Y, H, A, B = 1200.0, 0.25, 5.0, 30.0, 0.02, 0.02
mu = E / (2 * (1 + NU)); K = E / (3 * (1 - 2 * NU)); M = K + 4 * mu / 3
T = K + 4 * mu * H / (3 * (3 * mu + H)); OFF = 2 * mu * Y / (3 * mu + H)
def strain(x): return A + B * x
def alpha_full(x): return (2 * mu * strain(x) - Y) / (3 * mu + H)
def stress_full(x):
    e = strain(x); p = alpha_full(x)
    return np.array([T * e + OFF, K * e - 2 * mu * e / 3 + mu * p, K * e - 2 * mu * e / 3 + mu * p, 0, 0, 0])
S0 = stress_full(0.0)[0]; SLOPE = T * B
def front(f):
    if f == 0: return 1.0
    return min(max((M * Y / (2 * mu * f) - S0) / SLOPE, 0.0), 1.0)
def reference(x, f):
    r = front(f); ax = f * (S0 + SLOPE * x)
    eps = ax / M if ax <= M * Y / (2 * mu) else (ax - OFF) / T
    al = max((2 * mu * eps - Y) / (3 * mu + H), 0.0)
    ee = min(x, r)
    el = f * (S0 * ee + 0.5 * SLOPE * ee ** 2) / M
    pl = ((f * S0 - OFF) * (x - r) + 0.5 * f * SLOPE * (x * x - r * r)) / T if x > r else 0.0
    lat = K * eps - 2 * mu * eps / 3 + mu * al
    return el + pl, eps, al, np.array([ax, lat, lat, 0, 0, 0])

KEAST = [([0.25] * 4, 0.030283678097089)]
KEAST += [([0.0 if j == i else 1 / 3 for j in range(4)], 0.006026785714286) for i in range(4)]
KEAST += [([8 / 11 if j == i else 1 / 11 for j in range(4)], 0.011645249086029) for i in range(4)]
a_, b_ = 0.433449846426336, 0.066550153573664
for pair in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]:
    KEAST.append(([a_ if j in pair else b_ for j in range(4)], 0.010949141561386))
EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]

def mesh(cells, quad, lateral=None):
    lateral = cells if lateral is None else lateral
    n = cells + 1; m = lateral + 1
    nid = lambda x, y, z: x * m * m + y * m + z
    coords = [np.array([x / cells, y / lateral, z / lateral]) for x in range(n) for y in range(m) for z in range(m)]
    tets = []
    for x in range(cells):
        for y in range(lateral):
            for z in range(lateral):
                v = {k: nid(x + k[0], y + k[1], z + k[2]) for k in
                     [(0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1)]}
                for c in [[(0,0,0),(1,0,0),(1,1,0),(1,1,1)], [(0,0,0),(1,1,0),(0,1,0),(1,1,1)],
                          [(0,0,0),(0,1,0),(0,1,1),(1,1,1)], [(0,0,0),(0,1,1),(0,0,1),(1,1,1)],
                          [(0,0,0),(0,0,1),(1,0,1),(1,1,1)], [(0,0,0),(1,0,1),(1,0,0),(1,1,1)]]:
                    t = [v[k] for k in c]
                    P = [coords[i] for i in t]
                    if np.linalg.det(np.array([P[1]-P[0], P[2]-P[0], P[3]-P[0]])) < 0:
                        t[1], t[2] = t[2], t[1]
                    tets.append(t)
    if quad:
        emap = {}
        full = []
        for t in tets:
            mids = []
            for a, b in EDGES:
                key = tuple(sorted((t[a], t[b])))
                if key not in emap:
                    emap[key] = len(coords); coords.append((coords[t[a]] + coords[t[b]]) / 2)
                mids.append(emap[key])
            full.append(t + mids)
        tets = full
    return np.array(coords), [np.array(t) for t in tets]

def shape(L, quad):
    if not quad:
        return np.array(L), np.eye(4)
    N = [L[i] * (2 * L[i] - 1) for i in range(4)] + [4 * L[a] * L[b] for a, b in EDGES]
    dN = np.zeros((10, 4))
    for i in range(4): dN[i, i] = 4 * L[i] - 1
    for k, (a, b) in enumerate(EDGES):
        dN[4 + k, a] = 4 * L[b]; dN[4 + k, b] = 4 * L[a]
    return np.array(N), dN

def prepare(cells, quad, rule=KEAST, lateral=None):
    X, tets = mesh(cells, quad, lateral)
    pts = []  # (element index, N, G (nn,3), weight, position)
    for e, t in enumerate(tets):
        P = X[t[:4]]
        J = np.array([P[1] - P[0], P[2] - P[0], P[3] - P[0]]).T  # dx/dxi for L1..L3
        Jinv = np.linalg.inv(J); det = np.linalg.det(J)
        for L, w in rule:
            N, dNdL = shape(L, quad)
            dNdxi = dNdL[:, 1:] - dNdL[:, [0]]
            G = dNdxi @ Jinv
            pts.append((e, N, G, w * det, N[:len(t)] @ X[t] if quad else np.array(L) @ P))
    return X, tets, pts

def update(eps6, ep, al):
    tr_e = eps6 - ep
    tr = tr_e[:3].sum(); p = K * tr
    s = np.array([2*mu*(tr_e[i]-tr/3) for i in range(3)] + [mu*tr_e[i] for i in range(3,6)])
    s[2] = -s[0] - s[1]
    q = np.sqrt(1.5 * (s[:3] ** 2).sum() + 3 * (s[3:] ** 2).sum())
    Ct = np.zeros((6, 6))
    for i in range(3):
        for j in range(3): Ct[i, j] = K + 2*mu*(2/3 if i == j else -1/3)
    for i in range(3, 6): Ct[i, i] = mu
    cy = Y + H * al; scale = 1.0; ep2 = ep.copy(); al2 = al
    if q - cy > 0:
        den = 3 * mu + H; d = (q - cy) / den
        scale = (cy + H * d) / q; al2 = al + d
        ep2 = ep + np.array([1.5]*3 + [3.0]*3) * d * s / q
        r1 = -3*mu*(3*mu/den)*(cy/q)
        for i in range(6):
            for j in range(6):
                vol = K if (i < 3 and j < 3) else 0.0
                Ct[i, j] = vol + scale * (Ct[i, j] - vol) + r1 * s[i] / q * s[j] / q
    sig = scale * s + np.array([p]*3 + [0]*3)
    return sig, Ct, ep2, al2

def run(cells, quad, factors, verbose=True, rule=KEAST, lateral=None):
    X, tets, pts = prepare(cells, quad, rule, lateral)
    nn = len(X)
    fixed = np.isclose(X[:, 0], 0.0)
    free = ~fixed
    # loads
    F = np.zeros(nn)
    for e, N, G, w, x in pts:
        F[tets[e]] += N * (-T * B) * w
    # traction on x=1 faces: per tet faces with all corners at x=1
    tr = stress_full(1.0)[0]
    for t in tets:
        c = t[:4]
        on = [i for i in range(4) if np.isclose(X[c[i], 0], 1.0)]
        if len(on) == 3:
            P = X[c[on]]; area = 0.5 * np.linalg.norm(np.cross(P[1]-P[0], P[2]-P[0]))
            if not quad:
                F[c[on]] += tr * area / 3
            else:
                for k, (a, b) in enumerate(EDGES):
                    if a in on and b in on: F[t[4 + k]] += tr * area / 3
    nq = len(pts)
    ep = np.zeros((nq, 6)); al = np.zeros(nq)
    u = np.zeros(nn)
    out = []
    for f in factors:
        for it in range(50):
            Kg = np.zeros((nn, nn)); R = -f * F.copy()
            newep = np.zeros_like(ep); newal = np.zeros_like(al); sigs = np.zeros((nq, 6))
            for q, (e, N, G, w, x) in enumerate(pts):
                t = tets[e]; ue = u[t]
                g = G.T @ ue  # du/dx,dy,dz
                eps6 = np.array([g[0], 0, 0, g[1], 0, g[2]])
                sig, Ct, ep2, al2 = update(eps6, ep[q], al[q])
                Bm = np.zeros((6, len(t))); Bm[0] = G[:, 0]; Bm[3] = G[:, 1]; Bm[5] = G[:, 2]
                R[t] += Bm.T @ sig * w
                Kg[np.ix_(t, t)] += Bm.T @ Ct @ Bm * w
                newep[q] = ep2; newal[q] = al2; sigs[q] = sig
            rf = R[free]
            norm = np.linalg.norm(rf) / np.linalg.norm(f * F[free])
            if norm < 1e-13: break
            du = np.linalg.solve(Kg[np.ix_(free, free)], -rf)
            u[free] += du
        ep, al = newep, newal
        # errors
        ue_err = un = 0.0
        for e, N, G, w, x in pts:
            act = N @ u[tets[e]]; ex = reference(x[0], f)[0]
            ue_err += w * (act - ex) ** 2; un += w * ex ** 2
        se = sn = ae = an = 0.0
        for q, (e, N, G, w, x) in enumerate(pts):
            _, _, ral, rs = reference(x[0], f)
            se += w * ((sigs[q] - rs) ** 2).sum(); sn += w * (rs ** 2).sum()
            ae += w * (al[q] - ral) ** 2; an += w * alpha_full(x[0]) ** 2
        res = dict(f=f, it=it, disp=np.sqrt(ue_err / un), stress=np.sqrt(se / sn), alpha=np.sqrt(ae / an))
        if verbose: print(cells, quad, res, flush=True)
        out.append(res)
    return out, (X, tets, pts, u, al, ep)

if __name__ == "__main__":
    quad = sys.argv[1] == "tet10"
    cells = [int(c) for c in sys.argv[2].split(",")]
    for c in cells:
        run(c, quad, [i / 8 for i in range(1, 9)])
