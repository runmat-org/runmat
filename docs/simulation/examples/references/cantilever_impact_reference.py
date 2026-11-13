"""Euler-Bernoulli modal reference for `cantilever-impact/cantilever_impact.fea`.

A 1 m steel cantilever (0.1 m square section) carries a half-sine end force
F(t) = P sin(pi t / td) for 0 <= t <= td. The tip displacement is the sum of the
undamped modal Duhamel integrals of the first bending modes:

    w(L, t) = sum_n phi_n(L)^2 / (m L) * q_n(t),   q_n'' + w_n^2 q_n = F(t)

with phi_n normalized so that the integral of phi_n^2 over the length is L.
Prints the peak tip displacement, its time, and the tip displacement at sampled times.
"""
import math

P, td = 1000.0, 0.006
E, rho, b, L = 200e9, 7850.0, 0.1, 1.0
A, I = b * b, b**4 / 12.0
m = rho * A
betas = [1.875104068711961, 4.694091132974175, 7.854757438237613, 10.99554073487547, 14.13716839104647]


def phi_tip(beta_l):
    s = (math.sinh(beta_l) - math.sin(beta_l)) / (math.cosh(beta_l) + math.cos(beta_l))
    b_ = beta_l / L
    return (math.cosh(b_ * L) - math.cos(b_ * L)) - s * (math.sinh(b_ * L) - math.sin(b_ * L))


def q_half_sine(w, t):
    """Undamped response of q'' + w^2 q = sin(W t) on [0, td], free vibration after."""
    W = math.pi / td
    def forced(t):
        return (math.sin(W * t) - (W / w) * math.sin(w * t)) / (w * w - W * W)
    def forced_dot(t):
        return (W * math.cos(W * t) - W * math.cos(w * t)) / (w * w - W * W)
    if t <= td:
        return forced(t)
    q0, v0, s = forced(td), forced_dot(td), t - td
    return q0 * math.cos(w * s) + v0 / w * math.sin(w * s)


def tip(t):
    total = 0.0
    for beta_l in betas:
        w = beta_l**2 * math.sqrt(E * I / (m * L**4))
        total += phi_tip(beta_l) ** 2 / (m * L) * P * q_half_sine(w, t)
    return total


if __name__ == "__main__":
    f1 = betas[0] ** 2 / (2 * math.pi) * math.sqrt(E * I / (m * L**4))
    times = [k * 1e-5 for k in range(5001)]
    values = [tip(t) for t in times]
    k = max(range(len(values)), key=lambda i: abs(values[i]))
    print(f"first bending frequency {f1:.3f} Hz; static tip under P {P * L**3 / (3 * E * I) * 1e3:.5f} mm")
    print(f"peak |tip| {abs(values[k]) * 1e3:.5f} mm at {times[k] * 1e3:.2f} ms")
    for ms in (2, 4, 6, 8, 10, 15, 20, 30, 40, 50):
        print(f"t = {ms:4.1f} ms  tip = {tip(ms * 1e-3) * 1e3:+.5f} mm")
