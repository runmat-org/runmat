"""Kirchhoff reference for `square-plate.fea`: clamped 1 m square steel plate, 10 mm thick,
uniform 1 kPa pressure. Center deflection w = alpha p a^4 / D with alpha = 0.00126532
(Timoshenko and Woinowsky-Krieger, Table 35, nu = 0.3)."""
E, nu, t, a, p = 200e9, 0.3, 0.01, 1.0, 1000.0
D = E * t**3 / (12 * (1 - nu**2))
print(f"D = {D:.1f} N m; center deflection = {0.00126532 * p * a**4 / D * 1e3:.5f} mm")
