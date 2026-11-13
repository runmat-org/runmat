"""Euler-Bernoulli plane-frame reference for `portal-frame.fea`.

Fixed-base portal frame in the x-z plane: columns at x = 0 and x = 4 m, 3 m tall, joined by a
4 m girder; 100 mm x 100 mm steel section (E = 200 GPa, A = 0.01 m^2, I = 8.333e-6 m^4).
A 10 kN horizontal force in +x acts at the top of the left column. Cubic Euler-Bernoulli
beam elements reproduce the exact nodal response for nodal loads, so one element per member
gives the closed-form answer. Prints the joint displacements and rotations.
"""
import numpy as np

E, A, I = 200e9, 0.1 * 0.1, 0.1**4 / 12
H = 10e3
nodes = np.array([[0.0, 0.0], [0.0, 3.0], [4.0, 3.0], [4.0, 0.0]])
members = [(0, 1), (1, 2), (2, 3)]
K = np.zeros((12, 12))
for a, b in members:
    d = nodes[b] - nodes[a]
    L = np.hypot(*d)
    c, s = d / L
    k = np.array([
        [E * A / L, 0, 0, -E * A / L, 0, 0],
        [0, 12 * E * I / L**3, 6 * E * I / L**2, 0, -12 * E * I / L**3, 6 * E * I / L**2],
        [0, 6 * E * I / L**2, 4 * E * I / L, 0, -6 * E * I / L**2, 2 * E * I / L],
        [-E * A / L, 0, 0, E * A / L, 0, 0],
        [0, -12 * E * I / L**3, -6 * E * I / L**2, 0, 12 * E * I / L**3, -6 * E * I / L**2],
        [0, 6 * E * I / L**2, 2 * E * I / L, 0, -6 * E * I / L**2, 4 * E * I / L],
    ])
    T = np.zeros((6, 6))
    T[:2, :2] = T[3:5, 3:5] = [[c, s], [-s, c]]
    T[2, 2] = T[5, 5] = 1
    dofs = [3 * a, 3 * a + 1, 3 * a + 2, 3 * b, 3 * b + 1, 3 * b + 2]
    K[np.ix_(dofs, dofs)] += T.T @ k @ T
F = np.zeros(12)
F[3] = H
free = [3, 4, 5, 6, 7, 8]
u = np.zeros(12)
u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
for n, name in [(1, "top of loaded column"), (2, "top of far column")]:
    print(f"{name}: ux = {u[3*n]*1e3:.5f} mm, uz = {u[3*n+1]*1e3:+.5f} mm, rotation = {u[3*n+2]:+.6e} rad")
